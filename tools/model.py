"""1着確率モデルを学習し、調整期間・試験期間の予測を保存する。
使い方: python3 tools/model.py [--ex]
  --ex  展示タイムを使う（展示を見てから買う前提）

期間（日付で切る。未来のデータは学習に入れない）:
  履歴の助走  2023-01〜2023-06   … 選手の過去成績を貯めるだけ。学習に使わない
  学習        2023-07〜2025-06
  調整        2025-07〜2025-12   … 温度・λ・買い方の閾値はここだけで決める
  試験        2026-01〜          … 最後に一度だけ見る。見たあとで設定を変えたら、その結果は無効

出力: data/pred_<base|ex>.parquet（調整・試験期間の各艇の1着確率）
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
TRAIN = ("2023-07-01", "2025-06-30")
TUNE = ("2025-07-01", "2025-12-31")
TEST = ("2026-01-01", "2099-12-31")

BASE = ["lane", "venue", "rno", "distance", "race_class_c", "wind_speed", "wave",
        "grade_n", "nat_win", "nat_2", "loc_win", "loc_2", "motor_2", "boat_2", "age", "weight",
        "ser_n", "ser_avg",
        "hc_start", "hc_win_r", "hc_top2_r", "hc_top3_r", "ha_start", "ha_win_r", "ha_avg_rank",
        "ha_avg_st", "ha_fly"]
BASE += [f"{k}_{s}" for k in ["nat_win", "loc_win", "motor_2", "boat_2", "grade_n", "hc_win_r",
                              "hc_top2_r", "ha_avg_st", "ha_avg_rank", "age", "weight"] for s in ("rel", "rk")]
# 選手の戦い方・レース場との相性・隣の艇との関係（STYLE を外したモデルと比べて効果を測る）
STYLE = ["st_makuri", "st_sashi", "st_mz", "st_nuki", "st_nige", "st_attack", "st_wait",
         "hv_start", "hv_win_r", "hv_top2_r", "hv_gain",
         "in_st", "st_adv_in", "in_nige", "out_attack", "out_st_adv", "r_lane1_nige", "r_lane1_st",
         "attack_x_adv", "hv_win_r_rel", "hv_win_r_rk", "st_attack_rel", "st_attack_rk"]
# 調子・F持ち・STのばらつき・モーターの直近・風向き
FORM = ["h90_start", "h90_avg_rank", "form_delta", "f_hold", "ha_st_sd", "mo_start", "mo_top2_r", "mo_delta",
        "wind_dir_c", "mo_top2_r_rel", "mo_top2_r_rk", "h90_avg_rank_rel", "h90_avg_rank_rk",
        "ha_st_sd_rel", "ha_st_sd_rk"]
EX = ["exhibit", "exhibit_rel", "exhibit_rk", "ex_min_gap"]


def period(df, p):
    return df[(df["date"] >= p[0]) & (df["date"] <= p[1])]


def race_softmax(df, score, t=1.0):
    """レース内で 6艇の確率が合計1になるよう、スコアを温度 t 付きソフトマックスで確率にする。"""
    z = pd.Series(score.to_numpy() * t, index=df.index)
    e = np.exp(z - z.groupby(df["race_id"]).transform("max"))
    return e / e.groupby(df["race_id"]).transform("sum")


def race_nll(df, p):
    """1着艇に付けた確率の対数損失（1レース平均）。"""
    w = df["y_rank"] == 1
    return -np.log(np.clip(p[w], 1e-9, 1)).mean()


def lane_baseline(train, other):
    """枠×場だけの1着率（学習期間で集計）。超えるべき最低ライン。"""
    tab = train.groupby(["venue", "lane"])["y_rank"].apply(lambda s: (s == 1).mean())
    p = other.set_index(["venue", "lane"]).index.map(tab).to_numpy(dtype=float)
    p = pd.Series(p, index=other.index)
    return p / p.groupby(other["race_id"]).transform("sum")


def harville_exacta(p, lam):
    """6艇の1着確率 p から 2連単(i→j) の確率表を作る。2着側は p^λ で再配分（λ<1で穴が2着に来やすい）。"""
    q = p ** lam
    out = {}
    for i in range(6):
        rest = q.sum() - q[i]
        for j in range(6):
            if i != j and rest > 0:
                out[(i + 1, j + 1)] = p[i] * q[j] / rest
    return out


def fit_lambda(df, pcol):
    """調整期間の 1着→2着 の実績から λ を最尤で決める。"""
    races = []
    for rid, g in df.groupby("race_id", sort=False):
        if len(g) != 6:
            continue
        r = g.sort_values("lane")
        f = r.index[r["y_rank"] == 1]
        s = r.index[r["y_rank"] == 2]
        if len(f) != 1 or len(s) != 1:
            continue
        races.append((r[pcol].to_numpy(), int(r.loc[f[0], "lane"]) - 1, int(r.loc[s[0], "lane"]) - 1))

    def nll(lam):
        tot = 0.0
        for p, i, j in races:
            q = p ** lam
            tot -= np.log(max(q[j] / (q.sum() - q[i]), 1e-9))
        return tot / len(races)

    best = minimize_scalar(nll, bounds=(0.3, 1.5), method="bounded")
    return float(best.x), nll(1.0), float(best.fun)


W_FEATS = ["lane", "nat_win", "hc_win_r", "st_attack", "st_nige", "ha_avg_st"]


def second_rows(part, feats, winners):
    """『1着が winners[race] だったら』という仮定で、残り5艇の2着用の行を作る。
    winners: {race_id: [仮定する1着の枠,...]}。1着艇の性質（枠・強さ・攻め型か）を各行に付ける。"""
    w = part[["race_id"] + W_FEATS].rename(columns={k: "w_" + k for k in W_FEATS})
    keys = pd.DataFrame([(r, l) for r, ls in winners.items() for l in ls], columns=["race_id", "w_lane"])
    w = keys.merge(w, on=["race_id", "w_lane"])
    x = part.merge(w, on="race_id")
    x = x[x["lane"] != x["w_lane"]].copy()
    x["rel_w"] = x["lane"] - x["w_lane"]          # 1着艇から見た位置（差し・まくりの関係）
    x["adj_out_of_w"] = (x["rel_w"] == 1).astype(int)  # 1着艇のすぐ外
    x["key"] = x["race_id"] + "_" + x["w_lane"].astype(str)
    return x


S_EXTRA = ["w_" + k for k in W_FEATS] + ["rel_w", "adj_out_of_w"]


def train_second(tr, tu, feats, params):
    """2着モデル: 実際の1着艇を除いた5艇のうち、どれが2着かを学習する。"""
    def actual(part):
        win = part[part.y_rank == 1].groupby("race_id")["lane"].agg(list)
        win = win[win.map(len) == 1]
        return second_rows(part[part.race_id.isin(win.index)], feats, win.to_dict())

    xtr, xtu = actual(tr), actual(tu)
    f2 = feats + S_EXTRA
    cats = [c for c in ("venue", "race_class_c", "wind_dir_c") if c in f2]
    d1 = lgb.Dataset(xtr[f2], (xtr.y_rank == 2).astype(int), categorical_feature=cats)
    d2 = lgb.Dataset(xtu[f2], (xtu.y_rank == 2).astype(int), reference=d1)
    m2 = lgb.train(params, d1, 3000, valid_sets=[d2], callbacks=[lgb.early_stopping(100, verbose=False)])
    s = pd.Series(m2.predict(xtu[f2], num_iteration=m2.best_iteration, raw_score=True), index=xtu.index)
    xtu = xtu.assign(s=s, race_id_orig=xtu.race_id, race_id=xtu.key)
    t2 = minimize_scalar(lambda t: -np.log(np.clip(race_softmax(xtu, xtu["s"], t)[xtu.y_rank == 2], 1e-9, 1)).mean(),
                         bounds=(0.3, 3), method="bounded")
    nll2 = float(t2.fun)
    return m2, f2, float(t2.x), nll2


def exacta_table(pr, part, m2, f2, t2):
    """全30通りの2連単確率 = 1着確率 × 2着条件付き確率。"""
    winners = {r: list(range(1, 7)) for r in part.race_id.unique()}
    x = second_rows(part, None, winners)
    x["s"] = m2.predict(x[f2], num_iteration=m2.best_iteration, raw_score=True)
    x["p2"] = race_softmax(x.assign(race_id=x.key), x["s"], t2).to_numpy()
    pw = pr.set_index(["race_id", "lane"])["p_win"]
    x["p1"] = pw.reindex(list(zip(x.race_id, x.w_lane))).to_numpy()
    return pd.DataFrame({"race_id": x.race_id, "first": x.w_lane, "second": x.lane,
                         "p2_given": x.p2, "p_exacta": x.p1 * x.p2})


def exacta_nll(ex, part):
    """実際の 1着→2着 に付けた確率の対数損失。"""
    f = part[part.y_rank == 1].set_index("race_id")["lane"]
    s = part[part.y_rank == 2].set_index("race_id")["lane"]
    ok = f.index.intersection(s.index)
    f, s = f[~f.index.duplicated(keep=False)], s[~s.index.duplicated(keep=False)]
    ok = f.index.intersection(s.index)
    key = pd.MultiIndex.from_arrays([ok, f[ok].to_numpy(), s[ok].to_numpy()])
    p = ex.set_index(["race_id", "first", "second"])["p_exacta"].reindex(key)
    return float(-np.log(np.clip(p.to_numpy(), 1e-9, 1)).mean())


def harville_frame(pr, lam):
    rows = []
    for rid, g in pr.groupby("race_id", sort=False):
        p = g.sort_values("lane")["p_win"].to_numpy()
        if len(p) != 6:
            continue
        for (i, j), v in harville_exacta(p, lam).items():
            rows.append((rid, i, j, v))
    return pd.DataFrame(rows, columns=["race_id", "first", "second", "p_exacta"])


def main():
    use_ex = "--ex" in sys.argv
    no_style = "--nostyle" in sys.argv
    no_form = "--noform" in sys.argv
    feats = BASE + ([] if no_style else STYLE) + ([] if no_form else FORM) + (EX if use_ex else [])
    tag = ("ex" if use_ex else "base") + ("_nostyle" if no_style else "") + ("_noform" if no_form else "")
    use_course = "--course" in sys.argv
    if use_course:
        feats = feats + ["plan_course", "course_changed"]
        tag += "_course"
    df = pd.read_parquet(ROOT / ("features_course.parquet" if use_course else "features.parquet"))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.groupby("race_id")["lane"].transform("count") == 6]

    tr, tu, te = period(df, TRAIN), period(df, TUNE), period(df, TEST)
    if use_course:
        # 評価に使うのは本物の展示進入があるレースだけ（本番進入の代用はレース後の情報なので混ぜない）
        tu, te = tu[tu.ex_known == 1], te[te.ex_known == 1]
    print(f"[{tag}] 学習 {tr.race_id.nunique():,} / 調整 {tu.race_id.nunique():,} / 試験 {te.race_id.nunique():,} レース")

    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(tr[feats], (tr["y_rank"] == 1).astype(int), categorical_feature=[c for c in ("venue", "race_class_c", "wind_dir_c") if c in feats])
    dtu = lgb.Dataset(tu[feats], (tu["y_rank"] == 1).astype(int), reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dtu],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    print("木の本数", m.best_iteration)

    out = []
    for name, part in (("tune", tu), ("test", te)):
        if part.empty:
            continue
        s = pd.Series(m.predict(part[feats], num_iteration=m.best_iteration, raw_score=True),
                      index=part.index, name="s")
        part = part.assign(s=s)
        out.append(part.assign(period=name))
    pr = pd.concat(out)

    # 温度は調整期間だけで決める
    tun = pr[pr.period == "tune"]
    res = minimize_scalar(lambda t: race_nll(tun, race_softmax(tun, tun["s"], t)), bounds=(0.3, 3), method="bounded")
    temp = float(res.x)
    pr["p_win"] = race_softmax(pr, pr["s"], temp)
    pr["p_lane"] = lane_baseline(tr, pr)
    tun = pr[pr.period == "tune"]
    lam, nll1, nllb = fit_lambda(tun, "p_win")

    rep = {"tag": tag, "temperature": temp, "lambda": lam, "best_iter": m.best_iteration}
    for name in ("tune", "test"):
        g = pr[pr.period == name]
        if g.empty:
            continue
        top = g.loc[g.groupby("race_id")["p_win"].idxmax()]
        rep[name] = {
            "races": int(g.race_id.nunique()),
            "nll_model": race_nll(g, g["p_win"]), "nll_lane": race_nll(g, g["p_lane"]),
            "nll_uniform": float(np.log(6)),
            "top_pick_hit": float((top["y_rank"] == 1).mean()),
            "lane1_hit": float((g[g.lane == 1]["y_rank"] == 1).mean()),
        }
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"2着λ: {lam:.3f}（λ=1 の損失 {nll1:.4f} → {nllb:.4f}）")

    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("効いている特徴量 上位15:\n", (imp / imp.sum()).head(15).round(3).to_string())

    # 2着モデル（1着の条件付き）。Harville近似と、2連単の対数損失で比べる
    m2, f2, t2, nll2 = train_second(tr, tu, feats, params)
    exs = []
    for name in ("tune", "test"):
        part = pr[pr.period == name]
        if part.empty:
            continue
        src = tu if name == "tune" else te
        ex = exacta_table(pr[pr.period == name], src[src.race_id.isin(part.race_id)], m2, f2, t2)
        hv = harville_frame(part, lam)
        rep[name]["exacta_nll_second_model"] = exacta_nll(ex, part)
        rep[name]["exacta_nll_harville"] = exacta_nll(hv, part)
        exs.append(ex.assign(period=name))
        print(f"{name}: 2連単の対数損失  2着モデル {rep[name]['exacta_nll_second_model']:.4f} / "
              f"Harville {rep[name]['exacta_nll_harville']:.4f}（小さいほど良い）")
    rep["second_temperature"] = t2
    imp2 = pd.Series(m2.feature_importance("gain"), index=f2).sort_values(ascending=False)
    print("2着モデルで効いている特徴量 上位10:\n", (imp2 / imp2.sum()).head(10).round(3).to_string())
    pd.concat(exs).to_parquet(ROOT / f"exacta_{tag}.parquet")

    pr[["race_id", "date", "venue", "rno", "lane", "y_rank", "refund", "p_win", "p_lane", "period"]] \
        .to_parquet(ROOT / f"pred_{tag}.parquet")
    (ROOT / f"model_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
