"""モデル確率とオッズを比べて買い方を評価する。
使い方: python3 tools/bet.py [base|ex]

手順（すべての設定は調整期間だけで決め、試験期間は最後に1回だけ評価する）:
  1. 市場の確率 = 単勝オッズの逆数をレース内で正規化
  2. 混合確率 ∝ モデル^a × 市場^b   … a, b は調整期間の1着の対数尤度で決める
     （モデル単独より、市場の情報と組み合わせた方が確率として正確になる）
  3. 2連単・2連複 = 混合1着確率 × 2着モデルの条件付き確率（model.py の exacta_*.parquet）
  4. 期待値 = 混合確率 × 確定オッズ。期待値 >= θ の舟券を100円ずつ買う
     θ と券種は調整期間の回収率で選ぶ

注意: オッズは「確定オッズ」。締切前に見える値とは少し違うため、実戦では結果が少し悪くなりうる。
"""
import itertools, json, pathlib, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest import Book, evaluate, fmt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
THETAS = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]


def load(tag):
    pr = pd.read_parquet(ROOT / f"pred_{tag}.parquet")
    meta = json.loads((ROOT / f"model_{tag}.json").read_text())
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    od = pd.read_sql("SELECT * FROM odds", con)
    return pr, meta, od, con


def market_win(pr, od):
    t = od[od.bet_type == "tan"].assign(lane=lambda x: x.combo.astype(int))[["race_id", "lane", "odds"]]
    m = pr.merge(t, on=["race_id", "lane"], how="inner")
    m = m[m.groupby("race_id")["lane"].transform("count") == 6]
    m["q_mkt"] = (1 / m["odds"]) / (1 / m["odds"]).groupby(m["race_id"]).transform("sum")
    return m


def blend(m, a, b):
    z = a * np.log(np.clip(m["p_win"], 1e-6, 1)) + b * np.log(np.clip(m["q_mkt"], 1e-6, 1))
    e = np.exp(z - z.groupby(m["race_id"]).transform("max"))
    return e / e.groupby(m["race_id"]).transform("sum")


def fit_blend(tune):
    win = tune["y_rank"] == 1

    def nll(x):
        return -np.log(np.clip(blend(tune, x[0], x[1])[win], 1e-9, 1)).mean()

    r = minimize(nll, [1.0, 1.0], method="Nelder-Mead")
    mk = -np.log(np.clip(tune["q_mkt"][win], 1e-9, 1)).mean()
    md = -np.log(np.clip(tune["p_win"][win], 1e-9, 1)).mean()
    return r.x, {"nll_market": mk, "nll_model": md, "nll_blend": r.fun}


def candidates(m, lam, od, exacta=None):
    """各レースの全舟券（単勝・2連単・2連複）について、混合確率と確定オッズを並べる。
    exacta があれば 2着は 2着モデルの条件付き確率を使う（無ければ λ 付き Harville）。"""
    o = {(r, bt, c): x for r, bt, c, x in od[["race_id", "bet_type", "combo", "odds"]].itertuples(index=False)}
    cond = None
    if exacta is not None:
        cond = {(r, f, s): p for r, f, s, p in exacta[["race_id", "first", "second", "p2_given"]].itertuples(index=False)}
    rows = []
    for rid, g in m.groupby("race_id", sort=False):
        g = g.sort_values("lane")
        p = g["p_mix"].to_numpy()
        per = g["period"].iloc[0]
        for i in range(6):
            rows.append((rid, per, "tan", str(i + 1), p[i], o.get((rid, "tan", str(i + 1)))))
        q = p ** lam
        ex = {}
        for i, j in itertools.permutations(range(6), 2):
            if cond is not None and (rid, i + 1, j + 1) in cond:
                ex[(i, j)] = p[i] * cond[(rid, i + 1, j + 1)]
            else:
                ex[(i, j)] = p[i] * q[j] / (q.sum() - q[i])
            c = f"{i + 1}-{j + 1}"
            rows.append((rid, per, "nirentan", c, ex[(i, j)], o.get((rid, "nirentan", c))))
        for i, j in itertools.combinations(range(6), 2):
            c = f"{i + 1}-{j + 1}"
            rows.append((rid, per, "nirenpuku", c, ex[(i, j)] + ex[(j, i)], o.get((rid, "nirenpuku", c))))
    c = pd.DataFrame(rows, columns=["race_id", "period", "bet_type", "combo", "p", "odds"]).dropna()
    c["ev"] = c["p"] * c["odds"]
    return c


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    pr, meta, od, con = load(tag)
    m = market_win(pr, od)
    tune = m[m.period == "tune"]
    (a, b), info = fit_blend(tune)
    m["p_mix"] = blend(m, a, b)
    print(f"[{tag}] オッズのあるレース: 調整 {tune.race_id.nunique():,} / 試験 {m[m.period=='test'].race_id.nunique():,}")
    print(f"混合の重み モデル a={a:.2f} 市場 b={b:.2f}")
    print("1着の対数損失（調整期間）: 市場 {nll_market:.4f} / モデル {nll_model:.4f} / 混合 {nll_blend:.4f}".format(**info))

    book = Book(con)
    exf = ROOT / f"exacta_{tag}.parquet"
    cand = candidates(m, meta["lambda"], od, pd.read_parquet(exf) if exf.exists() else None)

    def run(df):
        return evaluate(book, ((r, bt, c, 100) for r, bt, c in df[["race_id", "bet_type", "combo"]].itertuples(index=False)))

    races_by = {p: sorted(m[m.period == p].race_id.unique()) for p in ("tune", "test")}
    print("\n■ 基準（同じレース集合で、何も考えずに買った場合）")
    for p in ("tune", "test"):
        for name, bt, combo in [("単勝 1", "tan", "1"), ("2連複 1=2 (C0)", "nirenpuku", "1-2")]:
            print(p, fmt(name, evaluate(book, ((r, bt, combo, 100) for r in races_by[p]))))

    print("\n■ モデルだけで選ぶ（オッズを見ない）")
    for p in ("tune", "test"):
        cp = cand[cand.period == p]
        for bt in ("tan", "nirenpuku"):
            top = cp[cp.bet_type == bt].loc[lambda x: x.groupby("race_id")["p"].idxmax()]
            print(p, fmt(f"{bt} 確率最大の1点", run(top)))

    print("\n■ 期待値で選ぶ（調整期間で θ を選ぶ）")
    grid = []
    for bt in ("tan", "nirentan", "nirenpuku"):
        for th in THETAS:
            sel = cand[(cand.period == "tune") & (cand.bet_type == bt) & (cand.ev >= th)]
            if len(sel) < 200:
                continue
            r = run(sel)
            grid.append((bt, th, r))
            print("tune", fmt(f"{bt} EV>={th}", r))
    if not grid:
        print("調整期間のデータが足りない")
        return
    # 選び方: 回収率の95%区間の下限が最も高いもの（偶然の当たりで選ばないため）
    bt, th, r = max(grid, key=lambda g: g[2]["ci95"][0])
    print(f"\n→ 採用: {bt} 期待値 >= {th}（調整期間の区間下限が最大）")
    sel = cand[(cand.period == "test") & (cand.bet_type == bt) & (cand.ev >= th)]
    if len(sel):
        rt = run(sel)
        print("test", fmt(f"{bt} EV>={th}", rt))
        # 最大払戻を除いた回収率
        pays = [book.settle(rr, b2, c, 100) for rr, b2, c in sel[["race_id", "bet_type", "combo"]].itertuples(index=False)]
        print(f"     最大払戻 {max(pays):,}円を除くと 回収率 {(sum(pays) - max(pays)) / (100 * len(pays)) * 100:.1f}%")


if __name__ == "__main__":
    main()
