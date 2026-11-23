"""モデル確率とオッズを比べて、期待値ベースの買い方を全券種・賭け金の配分ごとに評価する。
使い方: python3 tools/bet.py [tag] [--split dev|final]
  tag は model.py / walkforward.py の出力名（既定 base）
  --split dev   : ウォークフォワード予想を 調整2025-07〜09 / 試験2025-10〜12 で評価（封印前）
  --split final : 調整2025-07〜12 / 試験2026-（封印期間。walkforward.py --final の予想が必要。最後に1回だけ）

手順（設定はすべて調整期間だけで決め、試験期間は最後に1回だけ見る）:
  1. 市場の確率 = 単勝オッズの逆数をレース内で正規化
  2. 混合1着確率 ∝ モデル^a × 市場^b   … a, b は調整期間の1着の対数尤度で決める
  3. 2着 = 2着モデルの条件付き確率、3着 = λ3 で組み立て（strategies.py と同じ）
  4. 期待値 EV = 確率 × 確定オッズ。EV >= θ の舟券を買う
  5. 賭け金の配分を3通り比べる
       flat   : 1点100円
       equal  : 1レースの合計1,000円を、どれが当たっても払戻がほぼ同じになるよう 1/オッズ に比例配分
       kelly  : 資金10万円から開始し、1/4ケリー（f = (p·o − 1)/(o − 1) の1/4）で賭ける。
                1点は資金の2%まで。資金の推移と最大ドローダウンを出す

注意: 確定オッズを使うので、締切前に見えるオッズで買う実戦より少し甘く出る。
     拡連複・複勝は下限オッズで EV を計算する（控えめ）。
"""
import json, pathlib, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest import Book  # noqa: E402
import strategies as S  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
THETAS = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
BET_TYPES = ["tan", "fuku", "nirentan", "nirenpuku", "wide", "sanrentan", "sanrenpuku"]


def blend(pm, qm, a, b):
    z = a * np.log(np.clip(pm, 1e-6, 1)) + b * np.log(np.clip(qm, 1e-6, 1))
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def odds_matrix(od, rids, bt, names):
    o = od[od.bet_type == bt]
    idx = {r: n for n, r in enumerate(rids)}
    col = {c: k for k, c in enumerate(names)}
    M = np.full((len(rids), len(names)), np.nan)
    o = o[o.race_id.isin(idx) & o.combo.isin(col)]
    M[o.race_id.map(idx).to_numpy(), o.combo.map(col).to_numpy()] = o.odds.to_numpy()
    return M


def run_flat(back, choose, days):
    return S.evaluate(back, choose, days)


def run_equal(back, choose, O, days, budget=1000):
    """払戻を揃える配分。stake ∝ 1/odds、100円単位に丸め、合計 budget 円前後。"""
    ok = ~np.isnan(back).any(1) & choose.any(1)
    inv = np.zeros(len(back)); ret = np.zeros(len(back)); hit = np.zeros(len(back), bool)
    for n in np.where(ok)[0]:
        ks = np.where(choose[n])[0]
        w = 1 / O[n, ks]
        st = np.maximum(100, np.round(budget * w / w.sum() / 100) * 100)
        inv[n] = st.sum()
        r = back[n, ks] * st / 100
        ret[n] = r.sum()
        hit[n] = (back[n, ks] > 100).any()
    if inv.sum() == 0:
        return None
    df = pd.DataFrame({"d": days[ok], "i": inv[ok], "r": ret[ok]}).groupby("d").sum()
    rng = np.random.default_rng(0)
    b = rng.integers(0, len(df), (500, len(df)))
    boots = np.sort(df["r"].to_numpy()[b].sum(1) / df["i"].to_numpy()[b].sum(1))
    return {"races": int(ok.sum()), "race_hit": float(hit[ok].mean()), "roi": float(ret.sum() / inv.sum()),
            "ci_lo": float(boots[12]), "ci_hi": float(boots[487]), "invest": float(inv.sum())}


def run_kelly(back, choose, P, O, order, bank0=100_000, frac=0.25, cap=0.02):
    """時系列順に資金を動かす。同じレースの複数点は同時に賭ける（資金はレース開始時の値で計算）。"""
    bank, peak, mdd = bank0, bank0, 0.0
    curve = []
    for n in order:
        if np.isnan(back[n]).any() or not choose[n].any():
            continue
        ks = np.where(choose[n])[0]
        p, o = P[n, ks], O[n, ks]
        f = np.clip(frac * (p * o - 1) / np.maximum(o - 1, 1e-9), 0, cap)
        st = np.floor(f * bank / 100) * 100
        if st.sum() <= 0:
            continue
        bank += (back[n, ks] * st / 100).sum() - st.sum()
        peak = max(peak, bank)
        mdd = max(mdd, 1 - bank / peak)
        curve.append(bank)
        if bank < 100:
            break
    return {"final_bank": float(bank), "max_drawdown": float(mdd), "races": len(curve)}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else None
    rids, per, p1, P2, yr = S.build(tag)
    if split:
        # ウォークフォワードの予想を、オッズのある期間で区切り直す
        #   dev  : 調整 2025-07〜09 / 試験 2025-10〜12（封印前。何度使ってもよい）
        #   final: 調整 2025-07〜12 / 試験 2026-01〜（封印期間。最後に1回だけ）
        d = np.array([r[:8] for r in rids])
        if split == "dev":
            per = np.where((d >= "20250701") & (d <= "20250930"), "tune",
                           np.where((d >= "20251001") & (d <= "20251231"), "test", "x"))
        else:
            per = np.where((d >= "20250701") & (d <= "20251231"), "tune", np.where(d >= "20260101", "test", "x"))
        keep = per != "x"
        rids, per, p1, P2, yr = rids[keep], per[keep], p1[keep], P2[keep], yr[keep]
        tag = f"{tag}_{split}"
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    od = pd.read_sql("SELECT * FROM odds", con)
    tan = odds_matrix(od, rids, "tan", [str(i) for i in range(1, 7)])
    has = ~np.isnan(tan).any(1)
    rids, per, p1, P2, yr, tan = rids[has], per[has], p1[has], P2[has], yr[has], tan[has]
    q = (1 / tan) / (1 / tan).sum(1, keepdims=True)
    tune = per == "tune"
    w = yr == 1
    ok = w.sum(1) == 1

    def nll(x):
        pm = blend(p1[tune & ok], q[tune & ok], x[0], x[1])
        return -np.log(pm[w[tune & ok]]).mean()

    a, b = minimize(nll, [1.0, 1.0], method="Nelder-Mead").x
    pmix = blend(p1, q, a, b)
    print(f"[{tag}] オッズのあるレース: 調整 {tune.sum():,} / 試験 {(per == 'test').sum():,}")
    print(f"1着の対数損失（調整）: 市場 {-np.log(q[tune & ok][w[tune & ok]]).mean():.4f} / "
          f"モデル {-np.log(p1[tune & ok][w[tune & ok]]).mean():.4f} / 混合 {nll([a, b]):.4f}  (a={a:.2f}, b={b:.2f})")

    lam3 = S.fit_lambda3(pmix, P2, yr, tune)
    T = S.trifecta(pmix, P2, lam3)
    allc = S.combos(T, pmix, P2)
    book = Book(con)
    days = np.array([r[:8] for r in rids])
    order = np.argsort(rids)

    rows = []
    for bt in BET_TYPES:
        names, P = allc[bt]
        O = odds_matrix(od, rids, bt, names)
        if np.isnan(O).all():
            print(f"{bt}: オッズ未取得のため省略")
            continue
        back = S.settle_matrix(book, rids, names, bt)
        EV = P * O
        for th in THETAS:
            ch = (EV >= th) & ~np.isnan(O)
            for p in ("tune", "test"):
                m = per == p
                r = run_flat(back[m], ch[m], days[m])
                if r and r["races"] >= 30:
                    rows.append({"bet_type": bt, "theta": th, "stake": "flat", "period": p, **r})
                r = run_equal(back[m], ch[m], O[m], days[m])
                if r and r["races"] >= 30:
                    rows.append({"bet_type": bt, "theta": th, "stake": "equal", "period": p, **r})
                mo = order[np.isin(order, np.where(m)[0])]
                k = run_kelly(back, ch, P, O, mo)
                rows.append({"bet_type": bt, "theta": th, "stake": "kelly", "period": p, **k})
        print(bt, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / f"bets_{tag}.csv", index=False)
    if res.empty:
        print("評価できる組み合わせなし")
        return

    tu = res[(res.period == "tune") & (res.stake != "kelly")].sort_values("ci_lo", ascending=False)
    te = res[res.period == "test"].set_index(["bet_type", "theta", "stake"])
    print("\n■ 調整期間の上位15（回収率95%区間の下限順）→ 試験期間")
    print(f"{'券種':<11}{'EV≥':>5}{'配分':>6}{'R数':>6}{'的中':>7}{'回収率':>8}{'下限':>7} | {'試験R':>6}{'的中':>7}{'回収率':>8}{'下限':>7}")
    for _, r in tu.head(15).iterrows():
        k = (r.bet_type, r.theta, r.stake)
        t = te.loc[k] if k in te.index else None
        ts = f"{int(t.races):6d}{t.race_hit*100:6.1f}%{t.roi*100:7.1f}%{t.ci_lo*100:6.1f}%" if t is not None and not pd.isna(t.get("roi")) else ""
        print(f"{r.bet_type:<11}{r.theta:5.2f}{r.stake:>6}{int(r.races):6d}{r.race_hit*100:6.1f}%{r.roi*100:7.1f}%{r.ci_lo*100:6.1f}% | {ts}")

    print("\n■ 1/4ケリー（資金10万円から）調整期間の上位5 → 試験期間")
    ke = res[(res.stake == "kelly")]
    for _, r in ke[ke.period == "tune"].sort_values("final_bank", ascending=False).head(5).iterrows():
        t = ke[(ke.period == "test") & (ke.bet_type == r.bet_type) & (ke.theta == r.theta)].iloc[0]
        print(f"{r.bet_type:<11} EV≥{r.theta:.2f}  調整 {r.final_bank:>10,.0f}円 (最大DD {r.max_drawdown*100:.0f}%)"
              f"  → 試験 {t.final_bank:>10,.0f}円 (最大DD {t.max_drawdown*100:.0f}%)")


if __name__ == "__main__":
    main()
