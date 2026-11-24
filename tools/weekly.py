"""紙上記録（records/live/*_settled.csv）から「週いくら勝つには1点いくら賭けるか」と、そのときのぶれを出す。

使い方: python3 tools/weekly.py [週の目標額=5000]

選び方の偏りを避けるため、記録を日付で前半・後半に分ける。
  前半 … 券種×期待値の閾値の中から、回収率95%区間の下限が最も高い買い方を1つ選ぶ
  後半 … その買い方だけを、前半で決めた賭け金のまま当てはめて確かめる
後半でも回収率が100%を超えなければ、「週◯円」を狙える買い方は無い、が結論。

賭け金は「1点あたりの額を固定」。外した分を次で取り返す賭け方（倍賭けなど）は、
1回あたりの平均の損得を変えずにぶれと破産の危険だけを増やすので、ここでは扱わない。
"""
import pathlib, sys
import numpy as np
import pandas as pd

REC = pathlib.Path(__file__).resolve().parent.parent / "records" / "live"
THS = (1.0, 1.2, 1.5, 2.0)
JP = {"tan": "単勝", "fuku": "複勝", "wide": "拡連複", "nirenpuku": "2連複", "nirentan": "2連単",
      "sanrenpuku": "3連複", "sanrentan": "3連単"}


def load():
    fs = sorted(REC.glob("*_settled.csv"))
    if not fs:
        sys.exit("照合済みの記録がまだ無い")
    return pd.concat([pd.read_csv(f).assign(day=f.name[:8]) for f in fs])


def daily(c, bt, th):
    """1点100円のときの日ごとの 投資・払戻。記録の無い日は0（買う対象が無かった日）。"""
    g = c[(c.bet_type == bt) & (c.ev >= th)]
    days = sorted(c.day.unique())
    by = g.groupby("day").agg(inv=("back", "size"), back=("back", "sum")).reindex(days, fill_value=0)
    by["inv"] *= 100
    return by


def ci(by, rng, n=4000):
    idx = rng.integers(0, len(by), (n, len(by)))
    inv, back = by.inv.to_numpy(), by.back.to_numpy()
    r = back[idx].sum(1) / np.maximum(inv[idx].sum(1), 1)
    return np.percentile(r, 2.5) * 100, np.percentile(r, 97.5) * 100


def weeks(by, unit, rng, n=20000):
    """日を7日ずつ無作為に選んで1週間を作る。1点 unit 円のときの週の損益。"""
    pl = (by.back - by.inv).to_numpy() / 100 * unit
    return pl[rng.integers(0, len(pl), (n, 7))].sum(1)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    c = load()
    days = sorted(c.day.unique())
    half = days[len(days) // 2]
    a, b = c[c.day < half], c[c.day >= half]
    rng = np.random.default_rng(0)
    print(f"記録 {days[0]}〜{days[-1]}（{len(days)}日）。前半 {len(a.day.unique())}日で選び、後半 {len(b.day.unique())}日で確かめる\n")

    rows = []
    for bt in JP:
        for th in THS:
            by = daily(a, bt, th)
            if by.inv.sum() == 0:
                continue
            lo, hi = ci(by, rng)
            rows.append((bt, th, by.inv.sum() / 100, by.back.sum() / by.inv.sum() * 100, lo, hi))
    t = pd.DataFrame(rows, columns=["bt", "th", "点数", "回収率", "下限", "上限"]).sort_values("下限", ascending=False)
    print("■ 前半（選ぶ期間）上位5")
    for r in t.head(5).itertuples():
        print(f"  {JP[r.bt]:<4} EV≥{r.th:.1f}  {int(r.点数):5d}点  回収率 {r.回収率:6.1f}%（{r.下限:.0f}〜{r.上限:.0f}%）")

    best = t.iloc[0]
    ba = daily(a, best.bt, best.th)
    per_day_pts = ba.inv.mean() / 100
    edge = best.回収率 / 100 - 1
    print(f"\n選んだ買い方: {JP[best.bt]} 期待値{best.th:.1f}以上（1日平均 {per_day_pts:.1f}点）")
    if edge <= 0:
        print("前半の時点で回収率が100%以下。週いくら勝つ、を狙える買い方は無い")
        return
    unit = target / 7 / (per_day_pts * edge)
    unit = max(100, round(unit / 100) * 100)   # 舟券は100円単位
    print(f"前半の回収率どおりなら、週 {target:,}円 には 1点 {unit:,}円（1日の投資 約{per_day_pts * unit:,.0f}円）\n")

    bb = daily(b, best.bt, best.th)
    if bb.inv.sum() == 0:
        print("後半に該当する舟券が無い")
        return
    lo, hi = ci(bb, rng)
    roi = bb.back.sum() / bb.inv.sum() * 100
    w = weeks(bb, unit, rng)
    cum = ((bb.back - bb.inv) / 100 * unit).cumsum()
    print("■ 後半（確かめる期間）で、1点{:,}円を賭け続けたら".format(unit))
    print(f"  回収率 {roi:.1f}%（95%区間 {lo:.0f}〜{hi:.0f}%）  実際の損益 {cum.iloc[-1]:+,.0f}円 / {len(bb)}日")
    print(f"  途中の最大の落ち込み {(cum.cummax() - cum).max():,.0f}円")
    print(f"  1週間の損益（日を組み替えて2万通り）: 平均 {w.mean():+,.0f}円  中央 {np.median(w):+,.0f}円")
    print(f"    週{target:,}円以上勝つ {(w >= target).mean() * 100:.0f}%  / 負ける週 {(w < 0).mean() * 100:.0f}%"
          f"  / 悪い方から5% {np.percentile(w, 5):+,.0f}円")
    print("\n判定: 後半の回収率の95%区間の下限が100%を超えていなければ、実際に賭ける根拠にはならない")


if __name__ == "__main__":
    main()
