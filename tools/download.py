"""BOAT RACE公式（mbrace）から競走成績(K)・番組表(B)のLZHを日付範囲で取得する。
使い方: python3 tools/download.py 2023-01-01 2026-11-22
既に取得済みのファイルは飛ばす。サーバー負荷を避けるため1リクエストごとに待つ。"""
from concurrent.futures import ThreadPoolExecutor
import sys, time, datetime, pathlib, urllib.request, urllib.error

BASE = "https://www1.mbrace.or.jp/od2"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
WAIT = 0.7
WORKERS = 8

def fetch(kind, d):
    yymmdd = d.strftime("%y%m%d")
    url = f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{yymmdd}.lzh"
    out = ROOT / kind / f"{kind.lower()}{yymmdd}.lzh"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            out.write_bytes(data)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
        except Exception:
            pass
        time.sleep(2 ** attempt * 2)
    return "fail"

def job(kind, d):
    st = fetch(kind, d)
    if st != "skip":
        print(d, kind, st, flush=True)
        time.sleep(WAIT)

def main():
    start = datetime.date.fromisoformat(sys.argv[1])
    end = datetime.date.fromisoformat(sys.argv[2])
    days = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]
    # 1ファイル約10秒かかる（応答待ち）ため、同時8本までで取る
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for d in days:
            for kind in ("K", "B"):
                ex.submit(job, kind, d)

if __name__ == "__main__":
    main()
