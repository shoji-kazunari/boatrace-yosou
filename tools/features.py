"""1艇1行の特徴量テーブルを作る（data/features.parquet）。
使い方: python3 tools/features.py

未来の情報を混ぜないための約束:
  - 選手の過去成績（コース別・ST・着順）は「そのレース日の前日まで」のレースだけで集計する
  - 予想時点では実際の進入コースは分からないので、枠番どおりに入ると仮定してコース別成績を引く
  - 展示タイム(ex_*)は締切数分前に出る情報。展示を見てから買う場合だけ使える。
    モデルを「展示なし」「展示あり」に分けて比べること
  - 風・波(wind/wave)は成績ファイルの値＝レース時点。直前情報とほぼ同じだが厳密には締切後の値
"""
import pathlib, sqlite3
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
GRADE = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def load(con):
    races = pd.read_sql("SELECT * FROM races", con)
    cards = pd.read_sql("SELECT * FROM cards", con)
    res = pd.read_sql("SELECT race_id,lane,rank_raw,rank,toban,exhibit,course,st,st_raw FROM results", con)
    return races, cards, res


def series_stats(s):
    """今節成績 '1 3 25 4' のような着順の並び → (走数, 平均着順)。F/L等の記号は無視。"""
    digits = [int(c) for c in str(s) if c in "123456"]
    if not digits:
        return 0, np.nan
    return len(digits), float(np.mean(digits))


def history_feats(res, races):
    """選手の前日までの成績を、日付でmerge_asofできる累積表にする。"""
    r = res.merge(races[["race_id", "date"]], on="race_id")
    r["date"] = pd.to_datetime(r["date"])
    r["win"] = (r["rank"] == 1).astype(int)
    r["top2"] = (r["rank"] <= 2).astype(int)
    r["top3"] = (r["rank"] <= 3).astype(int)
    r["start"] = r["rank_raw"].str[0].isin(list("0SF")).astype(int)  # 出走した（欠場以外）
    r["fly"] = (r["rank_raw"] == "F").astype(int)
    r["st_ok"] = r["st"].where(r["st_raw"].fillna("").str.match(r"^\.?\d|^0"), np.nan)
    r["rank_f"] = r["rank"].fillna(6.5)

    # コース別（実際の進入コース）
    c = (r.dropna(subset=["course"]).assign(course=lambda x: x["course"].astype(int))
         .groupby(["toban", "course", "date"])[["start", "win", "top2", "top3"]].sum()
         .groupby(level=[0, 1]).cumsum().reset_index())
    # 全体
    a = (r.assign(st_n=r["st_ok"].notna().astype(int), st_sum=r["st_ok"].fillna(0))
         .groupby(["toban", "date"])[["start", "win", "top2", "fly", "st_n", "st_sum", "rank_f"]].sum()
         .groupby(level=0).cumsum().reset_index())
    return c, a


def asof_prev(left, right, by, prefix):
    """left の各行に、right の『前日まで』の累積値を付ける。"""
    right = right.copy()
    right["date"] = right["date"] + pd.Timedelta(days=1)  # 当日分を含めない
    right = right.sort_values("date")
    left = left.sort_values("date")
    out = pd.merge_asof(left, right, on="date", by=by, direction="backward")
    return out.rename(columns={k: prefix + k for k in right.columns if k not in by + ["date"]})


def main():
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    races, cards, res = load(con)
    races = races[races["race_id"].isin(res["race_id"].unique())]

    df = cards.merge(races, on="race_id").merge(
        res[["race_id", "lane", "rank_raw", "rank", "exhibit", "course"]], on=["race_id", "lane"], how="left")
    df = df[df["rank_raw"].notna()]
    df["date"] = pd.to_datetime(df["date"])
    df["grade_n"] = df["grade"].map(GRADE)
    ss = df["series"].map(series_stats)
    df["ser_n"] = [x[0] for x in ss]
    df["ser_avg"] = [x[1] for x in ss]
    df["plan_course"] = df["lane"]

    c, a = history_feats(res, races)
    c = c.rename(columns={"course": "plan_course"})
    df = asof_prev(df, c, ["toban", "plan_course"], "hc_")
    df = asof_prev(df, a, ["toban"], "ha_")
    for k in ["hc_start", "hc_win", "hc_top2", "hc_top3", "ha_start", "ha_win", "ha_top2", "ha_fly", "ha_st_n", "ha_st_sum", "ha_rank_f"]:
        df[k] = df[k].fillna(0)

    # 少ない標本を全体平均に寄せる（縮小）。コースごとの全体平均を事前分布にする
    K = 20
    prior = df.groupby("lane")["rank"].agg(lambda s: (s == 1).mean())
    prior2 = df.groupby("lane")["rank"].agg(lambda s: (s <= 2).mean())
    prior3 = df.groupby("lane")["rank"].agg(lambda s: (s <= 3).mean())
    df["hc_win_r"] = (df["hc_win"] + K * df["lane"].map(prior)) / (df["hc_start"] + K)
    df["hc_top2_r"] = (df["hc_top2"] + K * df["lane"].map(prior2)) / (df["hc_start"] + K)
    df["hc_top3_r"] = (df["hc_top3"] + K * df["lane"].map(prior3)) / (df["hc_start"] + K)
    df["ha_win_r"] = (df["ha_win"] + K / 6) / (df["ha_start"] + K)
    df["ha_avg_rank"] = (df["ha_rank_f"] + K * 3.5) / (df["ha_start"] + K)
    df["ha_avg_st"] = (df["ha_st_sum"] + 10 * 0.16) / (df["ha_st_n"] + 10)

    # レース内の相対値（同じレースの他艇と比べてどうか）
    g = df.groupby("race_id")
    for k in ["nat_win", "loc_win", "motor_2", "boat_2", "grade_n", "hc_win_r", "hc_top2_r",
              "ha_avg_st", "ha_avg_rank", "exhibit", "age", "weight"]:
        df[k + "_rel"] = df[k] - g[k].transform("mean")
        df[k + "_rk"] = g[k].rank(ascending=k in ("ha_avg_st", "ha_avg_rank", "exhibit"), method="average")
    df["ex_min_gap"] = df["exhibit"] - g["exhibit"].transform("min")

    df["y_rank"] = df["rank"].fillna(7).astype(int)
    df["refund"] = df["rank_raw"].str[0].isin(list("FLK"))
    df["race_class_c"] = df["race_class"].astype("category")
    df = df.sort_values(["date", "race_id", "lane"]).reset_index(drop=True)
    df.drop(columns=["name", "branch", "series", "early", "rank_raw", "kimarite", "deadline",
                     "race_class", "weather", "wind_dir"]).to_parquet(ROOT / "features.parquet")
    print(df.shape, df["date"].min().date(), df["date"].max().date())


if __name__ == "__main__":
    main()
