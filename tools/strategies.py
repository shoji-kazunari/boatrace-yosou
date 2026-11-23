"""オッズを使わない買い方（券種・点数・組み方・レース選び）を、確定払戻で総当たりに比べる。
使い方: python3 tools/strategies.py [tag]     # tag は model.py の出力名（既定 base）

確率の組み立て（model.py の出力を使う）:
  1着   p1(i)          = 1着モデル
  2着   p2(j|i)        = 2着モデル（1着の条件付き）
  3着   p3(k|i,j)      ∝ p1(k)^λ3（残り4艇で正規化。λ3 は調整期間の実績で決める）
  → 3連単 P(i,j,k) = p1(i) p2(j|i) p3(k|i,j)。ほかの券種はこれを足し合わせて作る
     2連単 = Σk / 2連複 = 2連単の両順 / 3連複 = 6順の和 / 拡連複(i,j) = 2艇とも3着以内
     単勝 = p1 / 複勝(i) = 2着以内（6艇立ては2着まで）

選び方: 調整期間で「回収率の95%区間の下限」が高い順に並べ、上位を試験期間で1回だけ確かめる。
         的中率（1レースで1点でも当たる率）も並べて出す。
"""
import itertools, json, pathlib, sqlite3, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest import Book  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
PERMS3 = list(itertools.permutations(range(6), 3))


def build(tag):
    pr = pd.read_parquet(ROOT / f"pred_{tag}.parquet")
    ex = pd.read_parquet(ROOT / f"exacta_{tag}.parquet")
    pr = pr[pr.groupby("race_id")["lane"].transform("count") == 6].sort_values(["race_id", "lane"])
    rids = pr["race_id"].drop_duplicates().to_numpy()
    R = len(rids)
    p1 = pr["p_win"].to_numpy().reshape(R, 6)
    yr = pr["y_rank"].to_numpy().reshape(R, 6)
    per = pr.groupby("race_id", sort=True)["period"].first().reindex(rids).to_numpy()
    idx = {r: n for n, r in enumerate(rids)}
    P2 = np.zeros((R, 6, 6))
    ex = ex[ex.race_id.isin(idx)]
    P2[ex.race_id.map(idx).to_numpy(), ex["first"].to_numpy() - 1, ex["second"].to_numpy() - 1] = ex["p2_given"].to_numpy()
    return rids, per, p1, P2, yr


def fit_lambda3(p1, P2, yr, mask):
    """実際の1・2着が与えられたときの3着艇の対数尤度で λ3 を決める。"""
    rows = []
    for n in np.where(mask)[0]:
        o = [np.where(yr[n] == k)[0] for k in (1, 2, 3)]
        if all(len(x) == 1 for x in o):
            rows.append((n, o[0][0], o[1][0], o[2][0]))
    rows = np.array(rows)

    def nll(lam):
        q = p1[rows[:, 0]] ** lam
        rest = q.sum(1) - q[np.arange(len(rows)), rows[:, 1]] - q[np.arange(len(rows)), rows[:, 2]]
        return -np.log(q[np.arange(len(rows)), rows[:, 3]] / rest).mean()

    r = minimize_scalar(nll, bounds=(0.2, 1.5), method="bounded")
    return float(r.x)


def trifecta(p1, P2, lam3, P3=None):
    """P3: R×6×6×6 の3着条件付き確率（3着モデル。無い組は NaN）。NaN の組は λ3 近似で埋める。"""
    R = len(p1)
    q = p1 ** lam3
    T = np.zeros((R, 6, 6, 6))
    for i, j, k in PERMS3:
        rest = q.sum(1) - q[:, i] - q[:, j]
        c = q[:, k] / rest
        if P3 is not None:
            c = np.where(np.isnan(P3[:, i, j, k]), c, P3[:, i, j, k])
        T[:, i, j, k] = p1[:, i] * P2[:, i, j] * c
    return T


def load_p3(tag, rids):
    f = ROOT / f"trifecta_{tag}.parquet"
    if not f.exists():
        return None
    t = pd.read_parquet(f)
    idx = {r: n for n, r in enumerate(rids)}
    t = t[t.race_id.isin(idx)]
    P3 = np.full((len(rids), 6, 6, 6), np.nan)
    P3[t.race_id.map(idx).to_numpy(), t["first"].to_numpy() - 1, t["second"].to_numpy() - 1,
       t["third"].to_numpy() - 1] = t["p3_given"].to_numpy()
    return P3


def combos(T, p1, P2):
    """券種ごとの (組み合わせ文字列リスト, 確率配列 R×N)。"""
    R = len(p1)
    ex = p1[:, :, None] * P2
    out = {}
    out["tan"] = ([str(i + 1) for i in range(6)], p1)
    top2 = ex.sum(2) + ex.sum(1)  # i が1着 or 2着
    out["fuku"] = ([str(i + 1) for i in range(6)], top2)
    c = list(itertools.permutations(range(6), 2))
    out["nirentan"] = ([f"{i+1}-{j+1}" for i, j in c], np.stack([ex[:, i, j] for i, j in c], 1))
    c = list(itertools.combinations(range(6), 2))
    out["nirenpuku"] = ([f"{i+1}-{j+1}" for i, j in c], np.stack([ex[:, i, j] + ex[:, j, i] for i, j in c], 1))
    wide = []
    for i, j in c:
        s = np.zeros(R)
        for a, b, d in PERMS3:
            if {i, j} <= {a, b, d}:
                s += T[:, a, b, d]
        wide.append(s)
    out["wide"] = ([f"{i+1}-{j+1}" for i, j in c], np.stack(wide, 1))
    out["sanrentan"] = ([f"{i+1}-{j+1}-{k+1}" for i, j, k in PERMS3], np.stack([T[:, i, j, k] for i, j, k in PERMS3], 1))
    c3 = list(itertools.combinations(range(6), 3))
    out["sanrenpuku"] = ([f"{i+1}-{j+1}-{k+1}" for i, j, k in c3],
                         np.stack([sum(T[:, a, b, d] for a, b, d in itertools.permutations(t)) for t in c3], 1))
    return out


def settle_matrix(book, rids, names, bt):
    """R×N の戻り額（100円あたり）。返還は100。"""
    M = np.zeros((len(rids), len(names)))
    for n, r in enumerate(rids):
        if r not in book.races:
            M[n] = np.nan
            continue
        for k, c in enumerate(names):
            M[n, k] = book.settle(r, bt, c, 100)
    return M


def evaluate(back, choose, days, rng_seed=0):
    """choose: R×N の買う(1)/買わない(0)。日単位ブートストラップで回収率の区間を出す。"""
    ok = ~np.isnan(back).any(1)
    ch = choose & ok[:, None]
    inv_r = ch.sum(1) * 100.0
    ret_r = np.where(ch, back, 0).sum(1)
    played = inv_r > 0
    if played.sum() == 0:
        return None
    hit_r = (np.where(ch, back, 0) > 100).any(1)
    df = pd.DataFrame({"d": days[played], "i": inv_r[played], "r": ret_r[played]}).groupby("d").sum()
    rng = np.random.default_rng(rng_seed)
    I, Rr = df["i"].to_numpy(), df["r"].to_numpy()
    b = rng.integers(0, len(df), (500, len(df)))
    boots = np.sort(Rr[b].sum(1) / I[b].sum(1))
    top = ret_r[played].max()
    return {"races": int(played.sum()), "points_per_race": float(ch[played].sum(1).mean()),
            "race_hit": float(hit_r[played].mean()), "roi": float(Rr.sum() / I.sum()),
            "ci_lo": float(boots[12]), "ci_hi": float(boots[487]),
            "roi_wo_max": float((Rr.sum() - top) / I.sum())}


def strategies(names, P, tune):
    """券種ごとの候補ルール: (名前, R×N の買う行列)。レース選びの境目は調整期間だけで決める。"""
    R, N = P.shape
    order = np.argsort(-P, 1)
    rank = np.empty_like(order)
    rank[np.arange(R)[:, None], order] = np.arange(N)[None, :]
    best = P.max(1)
    out = []
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if k >= N:
            continue
        topk = rank < k
        cum = np.take_along_axis(P, order[:, :k], 1).sum(1)  # 上位k点の合計確率＝当たる見込み
        out.append((f"上位{k}点 全レース", topk))
        for q in (0.5, 0.75, 0.9):  # 上位k点の合計確率が高い（堅い）レースだけ
            th = np.quantile(cum[tune], q)
            out.append((f"上位{k}点 堅い上位{round((1-q)*100)}%のレースだけ", topk & (cum >= th)[:, None]))
        th = np.quantile(cum[tune], 0.25)
        out.append((f"上位{k}点 荒れそうな下位25%のレースだけ", topk & (cum <= th)[:, None]))
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    rids, per, p1, P2, yr = build(tag)
    tune = per == "tune"
    lam3 = fit_lambda3(p1, P2, yr, tune)
    T = trifecta(p1, P2, lam3, load_p3(tag, rids))
    print(f"[{tag}] レース 調整 {tune.sum():,} / 試験 {(per == 'test').sum():,}   3着λ3={lam3:.3f}")
    book = Book(sqlite3.connect(ROOT / "boatrace.sqlite"))
    days = np.array([r[:8] for r in rids])
    allc = combos(T, p1, P2)
    results = []
    for bt, (names, P) in allc.items():
        back = settle_matrix(book, rids, names, bt)
        for name, ch in strategies(names, P, tune):
            for p in ("tune", "test"):
                m = per == p
                r = evaluate(back[m], ch[m], days[m])
                if r:
                    results.append({"bet_type": bt, "rule": name, "period": p, **r})
        print(bt, "done", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(ROOT / f"strategies_{tag}.csv", index=False)

    tu = res[res.period == "tune"].sort_values("ci_lo", ascending=False)
    te = res[res.period == "test"].set_index(["bet_type", "rule"])
    print("\n■ 調整期間の上位20（回収率95%区間の下限順）と、その試験期間の成績")
    print(f"{'券種':<11}{'ルール':<30}{'点/R':>5}{'的中':>7}{'回収率':>8}{'下限':>7} | {'試験 的中':>8}{'回収率':>8}{'下限':>7}{'最大除く':>8}")
    for _, r in tu.head(20).iterrows():
        t = te.loc[(r.bet_type, r.rule)] if (r.bet_type, r.rule) in te.index else None
        ts = f"{t.race_hit*100:7.1f}%{t.roi*100:7.1f}%{t.ci_lo*100:6.1f}%{t.roi_wo_max*100:7.1f}%" if t is not None else ""
        print(f"{r.bet_type:<11}{r.rule:<30}{r.points_per_race:5.1f}{r.race_hit*100:6.1f}%{r.roi*100:7.1f}%{r.ci_lo*100:6.1f}% | {ts}")

    print("\n■ 券種ごとの『的中率 と 回収率』の関係（調整期間・全レース買い）")
    for bt in allc:
        g = res[(res.period == "tune") & (res.bet_type == bt) & res.rule.str.endswith("全レース")]
        line = "  ".join(f"{int(r.points_per_race)}点:{r.race_hit*100:.0f}%/{r.roi*100:.0f}%" for _, r in g.iterrows())
        print(f"{bt:<11} {line}")


if __name__ == "__main__":
    main()
