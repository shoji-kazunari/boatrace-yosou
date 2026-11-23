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


def main():
    use_ex = "--ex" in sys.argv
    feats = BASE + (EX if use_ex else [])
    tag = "ex" if use_ex else "base"
    df = pd.read_parquet(ROOT / "features.parquet")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.groupby("race_id")["lane"].transform("count") == 6]

    tr, tu, te = period(df, TRAIN), period(df, TUNE), period(df, TEST)
    print(f"[{tag}] 学習 {tr.race_id.nunique():,} / 調整 {tu.race_id.nunique():,} / 試験 {te.race_id.nunique():,} レース")

    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(tr[feats], (tr["y_rank"] == 1).astype(int), categorical_feature=["venue", "race_class_c"])
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

    pr[["race_id", "date", "venue", "rno", "lane", "y_rank", "refund", "p_win", "p_lane", "period"]] \
        .to_parquet(ROOT / f"pred_{tag}.parquet")
    (ROOT / f"model_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
