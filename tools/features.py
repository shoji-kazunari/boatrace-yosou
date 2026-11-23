"""1艇1行の特徴量テーブルを作る（data/features.parquet）。
使い方: python3 tools/features.py [--course]
  --course  展示進入（直前情報）で進入コースを決める版を data/features_course.parquet に作る

未来の情報を混ぜないための約束:
  - 選手の過去成績（コース別・ST・着順）は「そのレース日の前日まで」のレースだけで集計する
  - 予想時点では実際の進入コースは分からないので、枠番どおりに入ると仮定してコース別成績を引く
  - 展示タイム(ex_*)は締切数分前に出る情報。展示を見てから買う場合だけ使える。
    モデルを「展示なし」「展示あり」に分けて比べること
  - 風・波(wind/wave)は成績ファイルの値＝レース時点。直前情報とほぼ同じだが厳密には締切後の値
"""
import pathlib, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
USE_COURSE = "--course" in sys.argv
GRADE = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def load(con):
    races = pd.read_sql("SELECT * FROM races", con)
    cards = pd.read_sql("SELECT * FROM cards", con)
    res = pd.read_sql("SELECT race_id,lane,rank_raw,rank,toban,motor,exhibit,course,st,st_raw FROM results", con)
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
    cols = ["start", "win", "top2", "fly", "st_n", "st_sum", "st_sq", "rank_f", "outer_start", "in_start",
            "k_nige", "k_makuri", "k_sashi", "k_mz", "k_nuki", "in_lost"]
    a = (r.assign(st_n=r["st_ok"].notna().astype(int), st_sum=r["st_ok"].fillna(0),
                  st_sq=r["st_ok"].fillna(0) ** 2)
         .groupby(["toban", "date"])[cols].sum()
         .groupby(level=0).cumsum().reset_index())
    # レース場別（その場の水面との相性）
    v = (r.groupby(["toban", "venue", "date"])[["start", "win", "top2"]].sum()
         .groupby(level=[0, 1]).cumsum().reset_index())
    # モーター（場×モーター番号）の成績。番組表の2連率は導入時からの累計なので、直近の調子を別に持つ
    mo = (r.dropna(subset=["motor"]).assign(motor_no=lambda x: x["motor"].astype(int))
          .groupby(["venue", "motor_no", "date"])[["start", "top2", "win"]].sum()
          .groupby(level=[0, 1]).cumsum().reset_index())
    return c, a, v, mo


def asof_prev(left, right, by, prefix):
    """left の各行に、right の『前日まで』の累積値を付ける。"""
    right = right.copy()
    right["date"] = right["date"] + pd.Timedelta(days=1)  # 当日分を含めない
    right = right.sort_values("date")
    left = left.sort_values("date")
    out = pd.merge_asof(left, right, on="date", by=by, direction="backward")
    return out.rename(columns={k: prefix + k for k in right.columns if k not in by + ["date"]})


def asof_window(left, right, by, prefix, days):
    """前日までの累積 − (days日前までの累積) ＝ 直近 days 日間の値を付ける。"""
    cols = [k for k in right.columns if k not in by + ["date"]]
    now = asof_prev(left[["date"] + by].reset_index(), right, by, "n_").set_index("index").sort_index()
    past_left = left[["date"] + by].assign(date=left["date"] - pd.Timedelta(days=days)).reset_index()
    past = asof_prev(past_left, right, by, "p_").set_index("index").sort_index()
    out = left.copy()
    for k in cols:
        out[prefix + k] = now["n_" + k].fillna(0) - past["p_" + k].fillna(0)
    return out


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
    # 予想に使う進入コース。
    #   既定: 枠なり（lane）
    #   --course: 直前情報の展示進入があればそれを使う。無いレース（学習期間）は本番の進入で代用する。
    #             展示進入と本番進入はほぼ一致するので学習の代用として使い、評価は本物の展示進入で行う
    df["plan_course"] = df["lane"]
    if USE_COURSE:
        df["plan_course"] = df["course"].fillna(df["lane"]).astype(int)
        try:
            bf = pd.read_sql("SELECT race_id, lane, ex_course FROM before WHERE ex_course IS NOT NULL", con)
            df = df.merge(bf, on=["race_id", "lane"], how="left")
            has = df.groupby("race_id")["ex_course"].transform("count") == 6
            df.loc[has, "plan_course"] = df.loc[has, "ex_course"].astype(int)
            df["ex_known"] = has.astype(int)  # 1 = 本物の展示進入。0 = 本番進入で代用（評価に使ってはいけない）
            df = df.drop(columns="ex_course")
            print("展示進入を使ったレース", int(has.sum() / 6))
        except Exception as e:  # before テーブルがまだ無い
            print("展示進入なし:", e)
    if "ex_known" not in df:
        df["ex_known"] = 0
    df["course_changed"] = (df["plan_course"] != df["lane"]).astype(int)

    c, a, v, mo = history_feats(res, races)
    c = c.rename(columns={"course": "plan_course"})
    df = df.reset_index(drop=True)
    # 直近の調子（90日）・F持ち（直近180日のフライング）・モーターの直近（120日）
    df = asof_window(df, a[["toban", "date", "start", "win", "top2", "rank_f"]], ["toban"], "h90_", 90)
    df = asof_window(df, a[["toban", "date", "fly"]], ["toban"], "h180_", 180)
    df = asof_window(df, mo, ["venue", "motor_no"], "mo_", 120)
    df = asof_prev(df, c, ["toban", "plan_course"], "hc_")
    df = asof_prev(df, a, ["toban"], "ha_")
    df = asof_prev(df, v, ["toban", "venue"], "hv_")
    for k in [x for x in df.columns if x[:3] in ("hc_", "ha_", "hv_")]:
        df[k] = df[k].fillna(0)

    # 少ない標本を全体平均に寄せる（縮小）。コースごとの全体平均を事前分布にする
    K = 20
    # 事前分布は学習期間の終わり(2025-06)までのデータだけで作る（調整・試験期間の結果を混ぜない）
    early = df[df["date"] < "2025-07-01"]
    prior = early.groupby("lane")["rank"].agg(lambda s: (s == 1).mean())
    prior2 = early.groupby("lane")["rank"].agg(lambda s: (s <= 2).mean())
    prior3 = early.groupby("lane")["rank"].agg(lambda s: (s <= 3).mean())
    df["hc_win_r"] = (df["hc_win"] + K * df["plan_course"].map(prior)) / (df["hc_start"] + K)
    df["hc_top2_r"] = (df["hc_top2"] + K * df["plan_course"].map(prior2)) / (df["hc_start"] + K)
    df["hc_top3_r"] = (df["hc_top3"] + K * df["plan_course"].map(prior3)) / (df["hc_start"] + K)
    df["ha_win_r"] = (df["ha_win"] + K / 6) / (df["ha_start"] + K)
    df["ha_avg_rank"] = (df["ha_rank_f"] + K * 3.5) / (df["ha_start"] + K)
    df["ha_avg_st"] = (df["ha_st_sum"] + 10 * 0.16) / (df["ha_st_n"] + 10)
    # STのばらつき（標準偏差）。平均が同じでも、安定して決める選手と博打の選手を分ける
    var = (df["ha_st_sq"] + 10 * (0.16 ** 2 + 0.04 ** 2)) / (df["ha_st_n"] + 10) - df["ha_avg_st"] ** 2
    df["ha_st_sd"] = np.sqrt(np.clip(var, 0, None))
    # 直近90日の調子（全期間の平均着順との差。マイナス＝最近好調）
    df["h90_avg_rank"] = (df["h90_rank_f"] + 10 * df["ha_avg_rank"]) / (df["h90_start"] + 10)
    df["form_delta"] = df["h90_avg_rank"] - df["ha_avg_rank"]
    df["f_hold"] = (df["h180_fly"] > 0).astype(int)   # F持ち（スタートを控えやすい）
    # モーター直近120日の2連率（縮小つき）と、番組表の累計2連率との差
    df["mo_top2_r"] = (df["mo_top2"] + 20 * 0.333) / (df["mo_start"] + 20)
    df["mo_delta"] = df["mo_top2_r"] * 100 - df["motor_2"]
    df["wind_dir_c"] = df["wind_dir"].astype("category")

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

    # 隣の艇との関係（並びはレース内で 予想進入コース順）
    df = df.sort_values(["race_id", "plan_course", "lane"]).reset_index(drop=True)
    g = df.groupby("race_id")
    df["in_st"] = g["ha_avg_st"].shift(1)                 # 1つ内側の艇の平均ST
    df["st_adv_in"] = df["in_st"] - df["ha_avg_st"]        # 内より速い(+)ならまくりやすい
    df["in_nige"] = g["st_nige"].shift(1)
    df["out_attack"] = g["st_attack"].shift(-1)           # 1つ外の艇の攻め型の強さ（自分が叩かれる危険）
    df["out_st_adv"] = df.groupby("race_id")["st_adv_in"].shift(-1)
    lane1 = df[df.plan_course == 1].drop_duplicates("race_id").set_index("race_id")
    df["r_lane1_nige"] = df["race_id"].map(lane1["st_nige"])   # 1号艇の逃げの強さ（全艇に効く）
    df["r_lane1_st"] = df["race_id"].map(lane1["ha_avg_st"])
    df["attack_x_adv"] = df["st_attack"] * df["st_adv_in"].fillna(0)  # 攻め型 × 内より速い

    # レース内の相対値（同じレースの他艇と比べてどうか）
    g = df.groupby("race_id")
    for k in ["nat_win", "loc_win", "motor_2", "boat_2", "grade_n", "hc_win_r", "hc_top2_r",
              "ha_avg_st", "ha_avg_rank", "exhibit", "age", "weight", "hv_win_r", "st_attack",
              "mo_top2_r", "h90_avg_rank", "ha_st_sd"]:
        df[k + "_rel"] = df[k] - g[k].transform("mean")
        df[k + "_rk"] = g[k].rank(ascending=k in ("ha_avg_st", "ha_avg_rank", "exhibit"), method="average")
    df["ex_min_gap"] = df["exhibit"] - g["exhibit"].transform("min")

    df["y_rank"] = df["rank"].fillna(7).astype(int)
    df["refund"] = df["rank_raw"].str[0].isin(list("FLK"))
    df["race_class_c"] = df["race_class"].astype("category")
    df = df.sort_values(["date", "race_id", "lane"]).reset_index(drop=True)
    df.drop(columns=["name", "branch", "series", "early", "rank_raw", "kimarite", "deadline",
                     "race_class", "weather", "wind_dir"], errors="ignore").to_parquet(ROOT / ("features_course.parquet" if USE_COURSE else "features.parquet"))
    print(df.shape, df["date"].min().date(), df["date"].max().date())


if __name__ == "__main__":
    main()
