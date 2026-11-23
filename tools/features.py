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
    r = res.merge(races[["race_id", "date", "venue", "kimarite"]], on="race_id")
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
    # 戦い方: 1着になったときの決まり手を、外枠(2〜6コース)からの出走数あたりで数える
    outer = (r["course"] >= 2).astype(int)
    won = r["rank"] == 1
    r["outer_start"] = outer * r["start"]
    r["in_start"] = (r["course"] == 1).astype(int) * r["start"]
    r["k_nige"] = (won & (r["kimarite"] == "逃げ")).astype(int)
    r["k_makuri"] = (won & (r["kimarite"] == "まくり")).astype(int)
    r["k_sashi"] = (won & (r["kimarite"] == "差し")).astype(int)
    r["k_mz"] = (won & (r["kimarite"] == "まくり差し")).astype(int)
    r["k_nuki"] = (won & (r["kimarite"] == "抜き")).astype(int)
    # 1コースで負けた理由（=内で弱いか）: 1コースで1着を逃した回数
    r["in_lost"] = ((r["course"] == 1) & (r["rank"] != 1)).astype(int) * r["start"]
    cols = ["start", "win", "top2", "fly", "st_n", "st_sum", "rank_f", "outer_start", "in_start",
            "k_nige", "k_makuri", "k_sashi", "k_mz", "k_nuki", "in_lost"]
    a = (r.assign(st_n=r["st_ok"].notna().astype(int), st_sum=r["st_ok"].fillna(0))
         .groupby(["toban", "date"])[cols].sum()
         .groupby(level=0).cumsum().reset_index())
    # レース場別（その場の水面との相性）
    v = (r.groupby(["toban", "venue", "date"])[["start", "win", "top2"]].sum()
         .groupby(level=[0, 1]).cumsum().reset_index())
    return c, a, v


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

    c, a, v = history_feats(res, races)
    c = c.rename(columns={"course": "plan_course"})
    df = asof_prev(df, c, ["toban", "plan_course"], "hc_")
    df = asof_prev(df, a, ["toban"], "ha_")
    df = asof_prev(df, v, ["toban", "venue"], "hv_")
    for k in [x for x in df.columns if x[:3] in ("hc_", "ha_", "hv_")]:
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

    # 戦い方（外枠からの出走あたりの決まり手別勝率、縮小つき）
    for k, base in [("makuri", 0.045), ("sashi", 0.035), ("mz", 0.03), ("nuki", 0.02)]:
        df[f"st_{k}"] = (df[f"ha_k_{k}"] + K * base) / (df["ha_outer_start"] + K)
    df["st_nige"] = (df["ha_k_nige"] + K * 0.55) / (df["ha_in_start"] + K)
    df["st_attack"] = df["st_makuri"] + df["st_mz"]   # 攻めて勝つ型
    df["st_wait"] = df["st_sashi"] + df["st_nuki"]    # 展開を待って勝つ型
    # レース場との相性（自前の履歴。当地勝率 loc_win とは別に、出走数つきで持つ）
    df["hv_win_r"] = (df["hv_win"] + K * df["ha_win_r"]) / (df["hv_start"] + K)
    df["hv_top2_r"] = (df["hv_top2"] + K * 2 * df["ha_win_r"]) / (df["hv_start"] + K)
    df["hv_gain"] = df["hv_win_r"] - df["ha_win_r"]  # その場で普段よりどれだけ勝つか

    # 隣の艇との関係（並びはレース内で lane 順）
    df = df.sort_values(["race_id", "lane"]).reset_index(drop=True)
    g = df.groupby("race_id")
    df["in_st"] = g["ha_avg_st"].shift(1)                 # 1つ内側の艇の平均ST
    df["st_adv_in"] = df["in_st"] - df["ha_avg_st"]        # 内より速い(+)ならまくりやすい
    df["in_nige"] = g["st_nige"].shift(1)
    df["out_attack"] = g["st_attack"].shift(-1)           # 1つ外の艇の攻め型の強さ（自分が叩かれる危険）
    df["out_st_adv"] = df.groupby("race_id")["st_adv_in"].shift(-1)
    lane1 = df[df.lane == 1].set_index("race_id")
    df["r_lane1_nige"] = df["race_id"].map(lane1["st_nige"])   # 1号艇の逃げの強さ（全艇に効く）
    df["r_lane1_st"] = df["race_id"].map(lane1["ha_avg_st"])
    df["attack_x_adv"] = df["st_attack"] * df["st_adv_in"].fillna(0)  # 攻め型 × 内より速い

    # レース内の相対値（同じレースの他艇と比べてどうか）
    g = df.groupby("race_id")
    for k in ["nat_win", "loc_win", "motor_2", "boat_2", "grade_n", "hc_win_r", "hc_top2_r",
              "ha_avg_st", "ha_avg_rank", "exhibit", "age", "weight", "hv_win_r", "st_attack"]:
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
