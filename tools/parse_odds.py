"""fetch_odds.py で保存したHTMLから確定オッズを読み、data/boatrace.sqlite の odds テーブルに入れる。
使い方: python3 tools/parse_odds.py

odds(race_id, bet_type, combo, odds)
  bet_type: tan / fuku(odds=下限) / nirentan / nirenpuku
  欠場などで売れていない組み合わせは入れない
"""
import gzip, pathlib, re, sqlite3, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
TD = re.compile(r"<td([^>]*)>(.*?)</td>", re.S)  # (属性文字列, 中身)


def tables(html):
    return re.findall(r"<table[^>]*>(.*?)</table>", html, re.S)


def rows(tb):
    body = tb.split("<tbody", 1)[1] if "<tbody" in tb else tb
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        yield [(c or "", re.sub(r"<[^>]+>", "", v).strip()) for c, v in TD.findall(tr)]


def odds_val(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_2tf(html):
    """2連単表・2連複表とも、列=1着(小さい方)の艇、行内の数字=相手艇。(艇番, オッズ)が対で並ぶ。"""
    out = []
    tbs = [t for t in tables(html) if "oddsPoint" in t]
    for bt, tb in zip(("nirentan", "nirenpuku"), tbs[:2]):
        for cells in rows(tb):
            for k in range(0, len(cells) - 1, 2):
                first = k // 2 + 1
                lab, val = cells[k][1], cells[k + 1]
                if not lab.isdigit() or "oddsPoint" not in val[0]:
                    continue
                o = odds_val(val[1])
                if o is None:
                    continue
                a, b = first, int(lab)
                combo = f"{a}-{b}" if bt == "nirentan" else f"{min(a, b)}-{max(a, b)}"
                out.append((bt, combo, o))
    return out


def parse_tf(html):
    out = []
    tbs = [t for t in tables(html) if "oddsPoint" in t]
    for bt, tb in zip(("tan", "fuku"), tbs[:2]):
        for cells in rows(tb):
            if not cells or not cells[0][1].isdigit():
                continue
            vals = [v for c, v in cells if "oddsPoint" in c]
            if not vals:
                continue
            o = odds_val(vals[0].split("-")[0])
            if o is not None:
                out.append((bt, cells[0][1], o))
    return out


def main():
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    con.execute("DROP TABLE IF EXISTS odds")
    con.execute("CREATE TABLE odds(race_id TEXT, bet_type TEXT, combo TEXT, odds REAL)")
    n = 0
    for p in sorted((ROOT / "raw" / "web").glob("*/*.html.gz")):
        date = p.parent.name
        venue, rno = int(p.name[:2]), int(p.name[2:4])
        page = p.name.split("_")[1].split(".")[0]
        html = gzip.decompress(p.read_bytes()).decode("utf8", "replace")
        recs = parse_2tf(html) if page == "odds2tf" else parse_tf(html)
        rid = f"{date}-{venue:02d}-{rno:02d}"
        con.executemany("INSERT INTO odds VALUES(?,?,?,?)", [(rid, *r) for r in recs])
        n += 1
    con.execute("CREATE INDEX IF NOT EXISTS ix_odds ON odds(race_id, bet_type)")
    con.commit()
    print(n, "pages")
    for r in con.execute("SELECT bet_type, COUNT(*), COUNT(DISTINCT race_id) FROM odds GROUP BY 1"):
        print(r)


if __name__ == "__main__":
    main()
