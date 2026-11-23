"""boatrace.jp から過去レースの確定オッズ（単勝・複勝 / 2連単・2連複）と直前情報を取得し、HTMLのまま保存する。
使い方: python3 tools/fetch_odds.py 2025-07-01 2026-11-22 6 [ページ,…]   # 6日おき
  ページ既定: oddstf,odds2tf,beforeinfo。3連単/3連複/拡連複は odds3t,odds3f,oddsk

- 対象レースは成績ファイル(K)に載っているものだけ。Kが無ければ先に取る
- 保存先: data/raw/web/<YYYYMMDD>/<場2桁><R2桁>_<page>.html.gz（取得済みは飛ばす）
- 1ページ約8.5秒（ほぼ応答待ち）なので同時8本で取る

注意: ここで取れるのは「確定オッズ」で、締切前に見えるオッズとは少し違う。
締切直前の売上の流入で動くため、確定オッズでの検証は実戦より少し甘く出る可能性がある。
"""
import datetime, gzip, pathlib, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import download, parse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
PAGES = ("oddstf", "odds2tf", "beforeinfo")  # beforeinfo = 直前情報（展示進入・展示ST・チルト・部品交換）
WORKERS = 8


def races_of(d):
    p = ROOT / "K" / f"k{d:%y%m%d}.lzh"
    if not p.exists():
        download.fetch("K", d)
    if not p.exists():
        return []
    out = []
    for venue, lines in parse.split_venues(parse.read_lzh(p), b"K"):
        for l in lines:
            m = re.match(rb"^\s+(\d+)R\s", l)
            if m and b"H" in l and b"m" in l:
                out.append((venue, int(m.group(1))))
    return out


def get(d, venue, rno, page):
    out = ROOT / "web" / f"{d:%Y%m%d}" / f"{venue:02d}{rno:02d}_{page}.html.gz"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.boatrace.jp/owpc/pc/race/{page}?rno={rno}&jcd={venue:02d}&hd={d:%Y%m%d}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            if page.startswith("odds") and b"oddsPoint" not in body:
                print("noodds", d, venue, rno, page, flush=True)
            out.write_bytes(gzip.compress(body))
            return
        except Exception as e:
            err = e
            time.sleep(5 * (attempt + 1))
    print("fail", d, venue, rno, page, err, flush=True)


def main():
    start = datetime.date.fromisoformat(sys.argv[1])
    end = datetime.date.fromisoformat(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    pages = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else PAGES
    d = start
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        while d <= end:
            rs = races_of(d)
            print(d, len(rs), "races", flush=True)
            list(ex.map(lambda a: get(d, *a), [(v, r, p) for v, r in rs for p in pages]))
            d += datetime.timedelta(days=step)


if __name__ == "__main__":
    main()
