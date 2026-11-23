"""買い方ルールの回収率を、全レースで計算する。

払戻金は成績ファイルの確定払戻（100円あたり）をそのまま使う。
当たった組み合わせの払戻さえ分かれば回収率は出せるので、オッズ履歴は不要。

返還の扱い（公式ルールに合わせた前提。検証の再現時に確認すること）:
  - 買い目に F(フライング)・L(出遅れ)・K(欠場) の艇を含む → 返還（投資を戻す）
  - S(失格) は返還なし
  - 着順が1件も無いレース（中止）は集計から除外
"""
import random
import sqlite3
import pathlib
from collections import defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "boatrace.sqlite"
ORDERED = {"tan", "nirentan", "sanrentan"}


def norm(bet_type, combo):
    if bet_type in ORDERED:
        return combo
    return "-".join(sorted(combo.split("-"), key=int))


class Book:
    """レースごとの払戻と返還艇を保持する。"""

    def __init__(self, con, date_from=None, date_to=None):
        where, args = [], []
        if date_from:
            where.append("r.date >= ?"); args.append(date_from)
        if date_to:
            where.append("r.date <= ?"); args.append(date_to)
        w = ("WHERE " + " AND ".join(where)) if where else ""
        self.pay = defaultdict(dict)
        for rid, bt, combo, p in con.execute(
                f"SELECT p.race_id,bet_type,combo,payout FROM payouts p JOIN races r USING(race_id) {w}", args):
            if combo == "特払い":
                continue
            self.pay[rid][(bt, norm(bt, combo))] = p
        self.refund_boats = defaultdict(set)
        self.races = set()
        for rid, lane, rr in con.execute(
                f"SELECT race_id,lane,rank_raw FROM results JOIN races r USING(race_id) {w}", args):
            self.races.add(rid)
            if rr and rr[0] in "FLK":
                self.refund_boats[rid].add(lane)
        self.races &= set(self.pay)  # 払戻の記録が無いレース（不成立など）は除く

    def settle(self, race_id, bet_type, combo, stake=100):
        """戻ってくる金額（払戻または返還）。"""
        boats = {int(x) for x in combo.split("-")}
        if boats & self.refund_boats[race_id]:
            return stake
        p = self.pay[race_id].get((bet_type, norm(bet_type, combo)))
        return p * stake // 100 if p else 0


def evaluate(book, bets):
    """bets: iterable of (race_id, bet_type, combo, stake)。
    日単位のブートストラップで回収率の95%区間も出す（同じ日のレースは独立でないため日単位で再標本化）。"""
    by_day = defaultdict(lambda: [0, 0])
    n = hits = 0
    inv = ret = 0
    for rid, bt, combo, stake in bets:
        if rid not in book.races:
            continue
        back = book.settle(rid, bt, combo, stake)
        n += 1
        inv += stake
        ret += back
        hits += back > stake
        d = by_day[rid[:8]]
        d[0] += stake
        d[1] += back
    days = list(by_day.values())
    rng = random.Random(0)
    boots = []
    for _ in range(1000):
        s = [days[rng.randrange(len(days))] for _ in days]
        i = sum(x[0] for x in s)
        boots.append(sum(x[1] for x in s) / i if i else 0)
    boots.sort()
    return {
        "bets": n, "hit_rate": hits / n if n else 0,
        "roi": ret / inv if inv else 0,
        "ci95": (boots[25], boots[975]) if boots else (0, 0),
        "invest": inv, "return": ret,
    }


def fmt(name, r):
    lo, hi = r["ci95"]
    return (f"{name:<28} {r['bets']:>8,}件  的中{r['hit_rate']*100:5.1f}%  "
            f"回収率{r['roi']*100:6.1f}%  (95%区間 {lo*100:5.1f}〜{hi*100:5.1f}%)")


def main():
    import sys
    con = sqlite3.connect(DB)
    book = Book(con, *(sys.argv[1:3] if len(sys.argv) >= 3 else (None, None)))
    races = sorted(book.races)
    print(f"対象 {len(races):,} レース  {races[0][:8]}〜{races[-1][:8]}\n")

    print("■ 何も考えない買い方（これが超えるべき基準）")
    for name, bt, combo in [
        ("単勝 1", "tan", "1"), ("単勝 2", "tan", "2"), ("単勝 6", "tan", "6"),
        ("2連複 1=2 (C0)", "nirenpuku", "1-2"), ("2連複 1=3", "nirenpuku", "1-3"),
        ("2連複 1=4", "nirenpuku", "1-4"), ("2連単 1-2", "nirentan", "1-2"),
        ("3連単 1-2-3", "sanrentan", "1-2-3"), ("3連複 1=2=3", "sanrenpuku", "1-2-3"),
    ]:
        print(fmt(name, evaluate(book, ((r, bt, combo, 100) for r in races))))

    print("\n■ 券種ごとの全買い（＝市場全体の払戻率）")
    import itertools
    lanes = range(1, 7)
    allc = {
        "tan": [str(a) for a in lanes],
        "nirentan": [f"{a}-{b}" for a, b in itertools.permutations(lanes, 2)],
        "nirenpuku": [f"{a}-{b}" for a, b in itertools.combinations(lanes, 2)],
        "sanrentan": [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(lanes, 3)],
        "sanrenpuku": [f"{a}-{b}-{c}" for a, b, c in itertools.combinations(lanes, 3)],
    }
    for bt, cs in allc.items():
        print(fmt(f"{bt} 全{len(cs)}点", evaluate(book, ((r, bt, c, 100) for r in races for c in cs))))


if __name__ == "__main__":
    main()
