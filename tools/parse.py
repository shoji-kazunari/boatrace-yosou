"""data/raw の K(競走成績)・B(番組表) LZH を読み、data/boatrace.sqlite に入れる。
使い方: python3 tools/parse.py
何度実行しても同じ結果になる（毎回作り直す）。

ファイル形式はBOAT RACE公式のダウンロードデータ（Shift_JIS固定長テキスト）。
列位置はバイト単位で揃っているので、デコード前のバイト列を切り出してから文字列にする。
"""
import pathlib, re, sqlite3, sys
import lhafile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "boatrace.sqlite"

ZEN = str.maketrans("０１２３４５６７８９ＲＨｍ：", "0123456789RHm:")

SCHEMA = """
CREATE TABLE races(
  race_id TEXT PRIMARY KEY,   -- YYYYMMDD-場コード-R番号
  date TEXT, venue INTEGER, rno INTEGER,
  race_class TEXT, distance INTEGER, weather TEXT, wind_dir TEXT,
  wind_speed INTEGER, wave INTEGER, kimarite TEXT, deadline TEXT
);
CREATE TABLE results(
  race_id TEXT, lane INTEGER, rank_raw TEXT, rank INTEGER, toban INTEGER,
  motor INTEGER, boat INTEGER, exhibit REAL, course INTEGER, st REAL, st_raw TEXT,
  PRIMARY KEY(race_id, lane)
);
CREATE TABLE payouts(
  race_id TEXT, bet_type TEXT, combo TEXT, payout INTEGER, popularity INTEGER
);
CREATE TABLE cards(
  race_id TEXT, lane INTEGER, toban INTEGER, name TEXT, age INTEGER, branch TEXT,
  weight INTEGER, grade TEXT,
  nat_win REAL, nat_2 REAL, loc_win REAL, loc_2 REAL,
  motor_no INTEGER, motor_2 REAL, boat_no INTEGER, boat_2 REAL,
  series TEXT, early TEXT,
  PRIMARY KEY(race_id, lane)
);
"""

BET_LABELS = {"単勝": "tan", "複勝": "fuku", "２連単": "nirentan", "２連複": "nirenpuku",
              "拡連複": "wide", "３連単": "sanrentan", "３連複": "sanrenpuku"}


def read_lzh(path):
    lf = lhafile.Lhafile(str(path))
    name = lf.namelist()[0]
    return lf.read(name)


def split_venues(data, kind):
    """1ファイルに全場ぶんが入っている。'NNKBGN'〜'NNKEND' で区切る。"""
    lines = data.replace(b"\r\n", b"\n").split(b"\n")
    cur, venue = None, None
    for l in lines:
        m = re.match(rb"^(\d\d)" + kind + rb"BGN", l)
        if m:
            venue, cur = int(m.group(1)), []
            continue
        if re.match(rb"^\d\d" + kind + rb"END", l):
            yield venue, cur
            cur = None
            continue
        if cur is not None:
            cur.append(l)


def num(s, f=float):
    try:
        return f(s)
    except (ValueError, TypeError):
        return None


def parse_k(date, data, cur):
    for venue, lines in split_venues(data, b"K"):
        i = 0
        n = len(lines)
        while i < n:
            t = lines[i].decode("cp932", "replace")
            m = re.match(r"^\s+(\d+)R\s+(\S+)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm", t)
            if not m:
                i += 1
                continue
            rno = int(m.group(1))
            race_id = f"{date}-{venue:02d}-{rno:02d}"
            hdr = lines[i + 1].decode("cp932", "replace")
            kimarite = hdr.split()[-1].replace("　", "") if "ﾚｰｽﾀｲﾑ" in hdr else None
            if kimarite == "ﾚｰｽﾀｲﾑ":  # 決まり手欄が空（不成立など）
                kimarite = None
            cur.execute("INSERT OR REPLACE INTO races(race_id,date,venue,rno,race_class,distance,weather,"
                        "wind_dir,wind_speed,wave,kimarite) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (race_id, date, venue, rno, m.group(2).replace("　", ""), int(m.group(3)),
                         m.group(4).replace("　", ""), m.group(5).replace("　", ""),
                         int(m.group(6)), int(m.group(7)), kimarite))
            i += 3
            # 着順行
            while i < n and lines[i].strip():
                row = lines[i]
                if len(row) < 30 or not re.match(rb"^\s+\S+\s+\d\s+\d{4}", row):
                    break
                rank_raw = row[0:4].decode().strip()
                lane = int(row[4:7])
                toban = int(row[8:12])
                rest = row[29:].decode("cp932", "replace").split()
                motor = num(rest[0], int) if len(rest) > 0 else None
                boat = num(rest[1], int) if len(rest) > 1 else None
                exhibit = num(rest[2]) if len(rest) > 2 else None
                course = num(rest[3], int) if len(rest) > 3 else None
                st_raw = rest[4] if len(rest) > 4 else None
                st = num(st_raw.lstrip("FL")) if st_raw else None
                # "00" が24件ほどある（意味不明の記録）。着順なしとして扱う
                rank = int(rank_raw) if rank_raw.isdigit() and int(rank_raw) >= 1 else None
                cur.execute("INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (race_id, lane, rank_raw, rank, toban, motor, boat, exhibit, course, st, st_raw))
                i += 1
            # 払戻行
            bet = None
            while i < n:
                t = lines[i].decode("cp932", "replace")
                if re.match(r"^\s+\d+R\s", t) or (t.strip() == "" and bet == "sanrenpuku"):
                    break
                s = t.strip()
                if not s:
                    i += 1
                    continue
                parts = s.split()
                if parts[0] in BET_LABELS:
                    bet = BET_LABELS[parts[0]]
                    parts = parts[1:]
                if bet is None:
                    i += 1
                    continue
                # 例: 1-2 1610 人気 8 / 複勝: 1 140 2 370 / 特払い 70 / 不成立
                j = 0
                while j < len(parts):
                    combo = parts[j]
                    if j + 1 < len(parts) and parts[j + 1].isdigit() and re.match(r"^[\d\-=]+$|^特払い$", combo):
                        pay = int(parts[j + 1])
                        pop = None
                        j += 2
                        if j + 1 < len(parts) and parts[j] == "人気":
                            pop = num(parts[j + 1], int)
                            j += 2
                        cur.execute("INSERT INTO payouts VALUES(?,?,?,?,?)", (race_id, bet, combo, pay, pop))
                    else:
                        j += 1
                i += 1


def parse_b(date, data, cur):
    for venue, lines in split_venues(data, b"B"):
        i, n = 0, len(lines)
        while i < n:
            t = lines[i].decode("cp932", "replace").translate(ZEN)
            m = re.match(r"^\s*(\d+)R\s", t)
            if not m or not re.search(r"H\d+m", t):
                i += 1
                continue
            rno = int(m.group(1))
            race_id = f"{date}-{venue:02d}-{rno:02d}"
            dl = re.search(r"締切予定(\d+:\d+)", t)
            cur.execute("INSERT OR IGNORE INTO races(race_id,date,venue,rno) VALUES(?,?,?,?)",
                        (race_id, date, venue, rno))
            if dl:
                cur.execute("UPDATE races SET deadline=? WHERE race_id=?", (dl.group(1), race_id))
            i += 1
            got = 0
            while i < n and got < 6:
                row = lines[i]
                if re.match(rb"^[1-6] \d{4}", row) and len(row) >= 65:
                    # ボートNOが3桁になると直前の2連率と空白なしで連結するので、必ず固定位置で切る
                    cols = [(24, 29), (29, 35), (35, 40), (40, 46), (46, 49), (49, 55), (55, 58), (58, 64)]
                    nums = [row[a:b].decode().strip() for a, b in cols]
                    cur.execute("INSERT OR REPLACE INTO cards VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        race_id, int(row[0:1]), int(row[2:6]),
                        row[6:14].decode("cp932", "replace").replace("　", ""),
                        num(row[14:16], int), row[16:20].decode("cp932", "replace"),
                        num(row[20:22], int), row[22:24].decode(),
                        num(nums[0]), num(nums[1]), num(nums[2]), num(nums[3]),
                        num(nums[4], int), num(nums[5]), num(nums[6], int), num(nums[7]),
                        row[65:77].decode("cp932", "replace"), row[77:].decode("cp932", "replace").strip()))
                    got += 1
                i += 1


def main():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    cur = con.cursor()
    files = sorted((RAW / "K").glob("*.lzh")) + sorted((RAW / "B").glob("*.lzh"))
    for k, p in enumerate(files):
        date = "20" + p.stem[1:]
        try:
            data = read_lzh(p)
            (parse_k if p.stem[0] == "k" else parse_b)(date, data, cur)
        except Exception as e:
            print("ERROR", p.name, e, file=sys.stderr)
        if k % 200 == 0:
            con.commit()
            print(k, "/", len(files), p.name, flush=True)
    con.commit()
    cur.execute("CREATE INDEX ix_pay ON payouts(race_id, bet_type)")
    cur.execute("CREATE INDEX ix_races_date ON races(date)")
    cur.execute("CREATE INDEX ix_cards_toban ON cards(toban)")
    con.commit()
    for tb in ("races", "results", "payouts", "cards"):
        print(tb, cur.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0])


if __name__ == "__main__":
    main()
