"""ウォークフォワード検証: 『その時点までのデータだけで作ったモデル』で次の期間を予想し、答え合わせする。
使い方: python3 tools/walkforward.py [--course] [--ex] [--nostyle] [--noform] [--step 3] [--final]

  開発用の予想期間  2024-07 〜 2025-12   … 試行錯誤はここの成績だけを見て行う
  封印期間          2026-01 〜           … --final を付けたときだけ予想する。最後に1回だけ

各区切り（既定3ヶ月ごと）で:
  学習     2023-07-01 〜 区切りの2ヶ月前
  検証     区切りの直前2ヶ月      … 木の本数・温度をここで決める（予想する期間は見ない）
  予想     区切りから step ヶ月

出力（strategies.py / bet.py にそのまま渡せる形）:
  data/pred_wf_<tag>.parquet, data/exacta_wf_<tag>.parquet
  period: 'tune' = 2024-07〜2025-03, 'test' = 2025-04〜2025-12, 'final' = 2026〜（--final時のみ）
  data/wf_log.csv に、試した設定と成績を1行ずつ追記する（試行錯誤の記録）
"""
import datetime, json, pathlib, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

ROOT = M.ROOT
DEV = ("2024-07-01", "2025-12-31")
FINAL = ("2026-01-01", "2099-12-31")


def months(start, end, step):
    d = pd.Timestamp(start)
    while d <= pd.Timestamp(end):
        e = d + pd.DateOffset(months=step) - pd.Timedelta(days=1)
        yield d, min(e, pd.Timestamp(end))
        d = d + pd.DateOffset(months=step)


def ds(x):
    return x.strftime("%Y-%m-%d")


def main():
    args = sys.argv[1:]
    step = int(args[args.index("--step") + 1]) if "--step" in args else 3
    use_course, use_ex = "--course" in args, "--ex" in args
    feats = M.BASE + ([] if "--nostyle" in args else M.STYLE) + ([] if "--noform" in args else M.FORM) \
        + (M.EX if use_ex else []) + (["plan_course", "course_changed"] if use_course else [])
    tag = "wf_" + ("ex" if use_ex else "base") + ("_nostyle" if "--nostyle" in args else "") \
        + ("_noform" if "--noform" in args else "") + ("_course" if use_course else "")
    df = pd.read_parquet(ROOT / ("features_course.parquet" if use_course else "features.parquet"))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.groupby("race_id")["lane"].transform("count") == 6]

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1)
    cats = [c for c in ("venue", "race_class_c", "wind_dir_c") if c in feats]
    spans = list(months(*DEV, step)) + (list(months(*FINAL, step)) if "--final" in args else [])
    preds, exs = [], []
    for s, e in spans:
        tr = M.period(df, ("2023-07-01", ds(s - pd.DateOffset(months=2) - pd.Timedelta(days=1))))
        va = M.period(df, (ds(s - pd.DateOffset(months=2)), ds(s - pd.Timedelta(days=1))))
        te = M.period(df, (ds(s), ds(e)))
        if te.empty:
            continue
        d1 = lgb.Dataset(tr[feats], (tr.y_rank == 1).astype(int), categorical_feature=cats)
        d2 = lgb.Dataset(va[feats], (va.y_rank == 1).astype(int), reference=d1)
        m = lgb.train(params, d1, 3000, valid_sets=[d2], callbacks=[lgb.early_stopping(100, verbose=False)])
        sv = pd.Series(m.predict(va[feats], num_iteration=m.best_iteration, raw_score=True), index=va.index)
        t = minimize_scalar(lambda x: M.race_nll(va, M.race_softmax(va, sv, x)), bounds=(0.3, 3), method="bounded").x
        st = pd.Series(m.predict(te[feats], num_iteration=m.best_iteration, raw_score=True), index=te.index)
        te = te.assign(p_win=M.race_softmax(te, st, t), p_lane=M.lane_baseline(tr, te))
        m2, f2, t2, _ = M.train_second(tr, va, feats, params)
        ex = M.exacta_table(te, te, m2, f2, t2)
        nll, nll_l = M.race_nll(te, te.p_win), M.race_nll(te, te.p_lane)
        print(f"{ds(s)}〜{ds(e)}  学習{tr.race_id.nunique():,}R  予想{te.race_id.nunique():,}R  "
              f"1着損失 {nll:.4f}（枠のみ {nll_l:.4f}）  2連単損失 {M.exacta_nll(ex, te):.4f}", flush=True)
        preds.append(te)
        exs.append(ex)

    pr = pd.concat(preds)
    pr["period"] = np.where(pr.date >= FINAL[0], "final", np.where(pr.date <= "2025-03-31", "tune", "test"))
    ex = pd.concat(exs).merge(pr[["race_id", "period"]].drop_duplicates(), on="race_id")
    pr[["race_id", "date", "venue", "rno", "lane", "y_rank", "refund", "p_win", "p_lane", "period"]] \
        .to_parquet(ROOT / f"pred_{tag}.parquet")
    ex.to_parquet(ROOT / f"exacta_{tag}.parquet")
    (ROOT / f"model_{tag}.json").write_text(json.dumps({"tag": tag, "lambda": 1.0}))

    dev = pr[pr.period != "final"]
    top = dev.loc[dev.groupby("race_id")["p_win"].idxmax()]
    row = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag, "args": " ".join(args),
           "n_feats": len(feats), "dev_races": dev.race_id.nunique(),
           "dev_nll_win": M.race_nll(dev, dev.p_win), "dev_nll_lane": M.race_nll(dev, dev.p_lane),
           "dev_nll_exacta": M.exacta_nll(ex[ex.period != "final"], dev),
           "dev_top_hit": float((top.y_rank == 1).mean())}
    if "--final" in args:
        fin = pr[pr.period == "final"]
        row["final_nll_win"] = M.race_nll(fin, fin.p_win)
        row["final_nll_exacta"] = M.exacta_nll(ex[ex.period == "final"], fin)
    log = ROOT / "wf_log.csv"
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(json.dumps(row, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
