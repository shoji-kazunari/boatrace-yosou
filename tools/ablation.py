"""特徴量のグループを1つずつ外して学習し、調整期間の精度がどれだけ落ちるかを見る。
使い方: python3 tools/ablation.py

落ち幅が大きいグループほど「効いている」。落ちない（むしろ良くなる）グループは不要。
判定は調整期間だけで行う（試験期間は見ない）。
"""
import pathlib, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

GROUPS = {
    "番組表（勝率・級別・モーター等）": ["grade_n", "nat_win", "nat_2", "loc_win", "loc_2", "motor_2", "boat_2",
                              "age", "weight"] + [f"{k}_{s}" for k in ["nat_win", "loc_win", "motor_2", "boat_2", "grade_n",
                                                                         "age", "weight"] for s in ("rel", "rk")],
    "コース別成績": ["hc_start", "hc_win_r", "hc_top2_r", "hc_top3_r"] + [f"{k}_{s}" for k in ["hc_win_r", "hc_top2_r"] for s in ("rel", "rk")],
    "全体成績・平均ST": ["ha_start", "ha_win_r", "ha_avg_rank", "ha_avg_st", "ha_fly"] + [f"{k}_{s}" for k in ["ha_avg_st", "ha_avg_rank"] for s in ("rel", "rk")],
    "今節成績": ["ser_n", "ser_avg"],
    "戦い方（決まり手）": ["st_makuri", "st_sashi", "st_mz", "st_nuki", "st_nige", "st_attack", "st_wait", "st_attack_rel", "st_attack_rk"],
    "レース場との相性": ["hv_start", "hv_win_r", "hv_top2_r", "hv_gain", "hv_win_r_rel", "hv_win_r_rk"],
    "隣の艇との関係": ["in_st", "st_adv_in", "in_nige", "out_attack", "out_st_adv", "r_lane1_nige", "r_lane1_st", "attack_x_adv"],
    "直近の調子": ["h90_start", "h90_avg_rank", "form_delta", "h90_avg_rank_rel", "h90_avg_rank_rk"],
    "F持ち・STのばらつき": ["f_hold", "ha_st_sd", "ha_st_sd_rel", "ha_st_sd_rk"],
    "モーター直近": ["mo_start", "mo_top2_r", "mo_delta", "mo_top2_r_rel", "mo_top2_r_rk"],
    "風・波・風向き": ["wind_speed", "wave", "wind_dir_c"],
}


def fit_eval(tr, tu, feats):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1)
    cats = [c for c in ("venue", "race_class_c", "wind_dir_c") if c in feats]
    d1 = lgb.Dataset(tr[feats], (tr.y_rank == 1).astype(int), categorical_feature=cats)
    d2 = lgb.Dataset(tu[feats], (tu.y_rank == 1).astype(int), reference=d1)
    m = lgb.train(params, d1, 2000, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])
    s = pd.Series(m.predict(tu[feats], num_iteration=m.best_iteration, raw_score=True), index=tu.index)
    r = minimize_scalar(lambda t: M.race_nll(tu, M.race_softmax(tu, s, t)), bounds=(0.3, 3), method="bounded")
    return float(r.fun)


def main():
    df = pd.read_parquet(M.ROOT / "features.parquet")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.groupby("race_id")["lane"].transform("count") == 6]
    tr, tu = M.period(df, M.TRAIN), M.period(df, M.TUNE)
    allf = M.BASE + M.STYLE + M.FORM
    full = fit_eval(tr, tu, allf)
    print(f"全部入り: 調整期間の1着対数損失 {full:.4f}  （{tu.race_id.nunique():,}レース）\n")
    rows = []
    for name, g in GROUPS.items():
        f = [x for x in allf if x not in g]
        v = fit_eval(tr, tu, f)
        rows.append((name, v - full))
        print(f"{name:<24} 外すと {v:.4f}  （悪化 {v - full:+.4f}）", flush=True)
    print("\n効いている順:")
    for name, d in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:<24} {d:+.4f}")


if __name__ == "__main__":
    main()
