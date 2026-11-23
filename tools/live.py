"""締切前のオッズで「買ったつもり」を記録する紙上検証（お金は賭けない）。
確定オッズでの検証は、締切後にしか分からないオッズで選んでいる偏りがある。
実戦で見える「締切数分前のオッズ」でも期待値の高い舟券が当たるのかを、ここで確かめる。

使い方:
  python3 tools/live.py train              # 前日までの全データで学習し data/live/ に保存（1日1回）
  python3 tools/live.py prepare [YYYYMMDD]  # 当日の番組表を取得し、当日分の特徴量を作る
  python3 tools/live.py run [YYYYMMDD]      # 各レースの締切5分前に直前情報とオッズを取り、記録する
  python3 tools/live.py settle YYYYMMDD     # 翌日以降に結果と確定オッズで答え合わせ

記録: data/live/<日付>_cands.csv … 期待値1以上の全候補（券種・組・確率・締切前オッズ・取得時刻）
      data/raw/live/<日付>/…      … 取得したページそのもの

モデル: 1着=レース単位の多項ロジット、2着・3着=条件付きモデル。特徴量は BASE+STYLE+FORM+EX から
       風向き（締切前に正確に取れない）を除いたもの。展示タイム・風速・波高は直前情報ページの値を使う。
"""
import datetime, gzip, itertools, json, pathlib, re, sqlite3, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import walkforward as W  # noqa: E402
import parse_odds as PO  # noqa: E402
import strategies as S  # noqa: E402
import download  # noqa: E402

ROOT = HERE.parent / "data"
LIVE = ROOT / "live"
JST = datetime.timezone(datetime.timedelta(hours=9))
FEATS = [f for f in M.BASE + M.STYLE + M.FORM + M.EX if f != "wind_dir_c"]
PAGES = ("beforeinfo", "oddstf", "odds2tf", "odds3t")
LEAD_MIN = 5          # 締切の何分前に取るか
BLEND = (0.68, 0.38)  # モデル^a × 市場^b（bet.py dev で確定オッズに当てはめた値。締切前オッズでは要再推定）
LAM3 = 0.45


def today():
    return datetime.datetime.now(JST).strftime("%Y%m%d")


def sh(*a):
    subprocess.run([sys.executable, *a], check=True, cwd=HERE.parent)


# ---------- 学習 ----------
def train():
    LIVE.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "features.parquet")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.groupby("race_id")["lane"].transform("count") == 6].sort_values(["race_id", "lane"])
    last = df["date"].max()
    cut = (pd.Timestamp(last) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    tr, va = M.period(df, ("2023-07-01", cut)), df[df.date > cut]
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1)
    cats = [c for c in ("venue", "race_class_c") if c in FEATS]
    trw, vaw = W.one_winner(tr), W.one_winner(va)
    d1 = lgb.Dataset(trw[FEATS], (trw.y_rank == 1).astype(int), categorical_feature=cats, free_raw_data=False)
    d2 = lgb.Dataset(vaw[FEATS], (vaw.y_rank == 1).astype(int), reference=d1)
    m1 = lgb.train(dict(params, objective=W.softmax_obj, metric="None"), d1, 4000, valid_sets=[d2],
                   feval=W.softmax_eval, callbacks=[lgb.early_stopping(100, verbose=False)])
    sv = pd.Series(m1.predict(va[FEATS], num_iteration=m1.best_iteration, raw_score=True), index=va.index)
    t1 = minimize_scalar(lambda x: M.race_nll(va, M.race_softmax(va, sv, x)), bounds=(0.3, 3), method="bounded").x
    m2, f2, t2, _ = M.train_second(tr, va, FEATS, params)
    m3, f3, t3, _ = M.train_third(tr, va, FEATS, params)
    m1.save_model(str(LIVE / "m1.txt"), num_iteration=m1.best_iteration)
    m2.save_model(str(LIVE / "m2.txt"), num_iteration=m2.best_iteration)
    m3.save_model(str(LIVE / "m3.txt"), num_iteration=m3.best_iteration)
    cfg = {"trained_through": last, "t1": float(t1), "t2": t2, "t3": t3, "f2": f2, "f3": f3, "feats": FEATS,
           "cats": {c: list(df[c].cat.categories) for c in cats if hasattr(df[c], "cat")}}
    (LIVE / "config.json").write_text(json.dumps(cfg, ensure_ascii=False))
    print("学習完了", last, "温度", t1, t2, t3)


# ---------- 準備 ----------
def prepare(day):
    d = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:]))
    for k in ("K", "B"):
        download.fetch(k, d - datetime.timedelta(days=1))
    download.fetch("B", d)
    sh("tools/parse.py")
    sh("tools/features.py", "--keep-open", day, "--out", f"live/feats_{day}.parquet")
    f = pd.read_parquet(LIVE / f"feats_{day}.parquet")
    f = f[f["date"].dt.strftime("%Y%m%d") == day]
    f.to_parquet(LIVE / f"feats_{day}.parquet")
    print("当日のレース", f.race_id.nunique())


# ---------- 記録 ----------
def fetch(day, venue, rno, page):
    url = f"https://www.boatrace.jp/owpc/pc/race/{page}?rno={rno}&jcd={venue:02d}&hd={day}"
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read()
    out = ROOT / "raw" / "live" / day / f"{venue:02d}{rno:02d}_{page}_{datetime.datetime.now(JST):%H%M%S}.html.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(body))
    return body.decode("utf8", "replace")


def weather(html):
    h = re.sub(r"\s+", " ", html)
    w = re.search(r"風速</span> <span[^>]*>(\d+)m", h)
    v = re.search(r"波高</span> <span[^>]*>(\d+)cm", h)
    return (int(w.group(1)) if w else np.nan), (int(v.group(1)) if v else np.nan)


def predict_race(g, before_html, cfg, m1, m2, m3):
    """g: 1レース6行の特徴量。直前情報で展示タイム・風・波を埋めてから予想する。"""
    g = g.sort_values("lane").copy()
    bf = {r[0]: r for r in PO.parse_before(before_html)}
    g["exhibit"] = [bf.get(l, (None,) * 9)[8] for l in g.lane]
    ws, wv = weather(before_html)
    g["wind_speed"], g["wave"] = ws, wv
    g["exhibit_rel"] = g["exhibit"] - g["exhibit"].mean()
    g["exhibit_rk"] = g["exhibit"].rank(method="average")
    g["ex_min_gap"] = g["exhibit"] - g["exhibit"].min()
    s = pd.Series(m1.predict(g[cfg["feats"]], raw_score=True), index=g.index)
    g["p_win"] = M.race_softmax(g, s, cfg["t1"])
    ex = M.exacta_table(g, g, m2, cfg["f2"], cfg["t2"])
    tri = M.trifecta_table(g, ex.assign(p_exacta=ex.p_exacta), m3, cfg["f3"], cfg["t3"], topk=30)
    p1 = g["p_win"].to_numpy()[None, :]
    P2 = np.zeros((1, 6, 6))
    P2[0, ex["first"] - 1, ex["second"] - 1] = ex["p2_given"]
    P3 = np.full((1, 6, 6, 6), np.nan)
    P3[0, tri["first"] - 1, tri["second"] - 1, tri["third"] - 1] = tri["p3_given"]
    return g, p1, P2, P3


def run(day):
    cfg = json.loads((LIVE / "config.json").read_text())
    m1, m2, m3 = (lgb.Booster(model_file=str(LIVE / f"m{k}.txt")) for k in (1, 2, 3))
    feats = pd.read_parquet(LIVE / f"feats_{day}.parquet")
    for c, cats in cfg["cats"].items():
        feats[c] = pd.Categorical(feats[c], categories=cats)
    races = sqlite3.connect(ROOT / "boatrace.sqlite").execute(
        "SELECT race_id, venue, rno, deadline FROM races WHERE date=? AND deadline IS NOT NULL", (day,)).fetchall()
    todo = {}
    for rid, v, r, dl in races:
        hh, mm = map(int, dl.split(":"))
        t = datetime.datetime(int(day[:4]), int(day[4:6]), int(day[6:]), hh, mm, tzinfo=JST)
        todo[rid] = (v, r, t)
    out = LIVE / f"{day}_cands.csv"
    done = set(pd.read_csv(out).race_id) if out.exists() else set()
    print(f"{day}: {len(todo)}レース（記録済み {len(done)}）", flush=True)
    pool = ThreadPoolExecutor(8)
    while True:
        now = datetime.datetime.now(JST)
        due = [rid for rid, (v, r, t) in todo.items()
               if rid not in done and t - datetime.timedelta(minutes=LEAD_MIN) <= now < t - datetime.timedelta(minutes=1)]
        for rid in due:
            done.add(rid)
            pool.submit(record, day, rid, *todo[rid], feats, cfg, m1, m2, m3, out)
        if all(t < now for _, _, t in todo.values()):
            break
        time.sleep(15)
    pool.shutdown(wait=True)


def record(day, rid, venue, rno, deadline, feats, cfg, m1, m2, m3, out):
    try:
        got = {}
        with ThreadPoolExecutor(len(PAGES)) as ex:
            for p, h in zip(PAGES, ex.map(lambda p: fetch(day, venue, rno, p), PAGES)):
                got[p] = h
        snap = datetime.datetime.now(JST)
        g = feats[feats.race_id == rid]
        if len(g) != 6:
            return
        g, p1, P2, P3 = predict_race(g, got["beforeinfo"], cfg, m1, m2, m3)
        odds = {}
        for bt, c, o in PO.parse_tf(got["oddstf"]) + PO.parse_2tf(got["odds2tf"]) + PO.parse_3(got["odds3t"], "sanrentan"):
            if o and o > 0:
                odds[(bt, c)] = o
        tan = np.array([odds.get(("tan", str(i)), np.nan) for i in range(1, 7)])
        q = np.where(np.isnan(tan), 1e-4, 1 / tan)
        q = (q / q.sum())[None, :]
        a, b = BLEND
        z = a * np.log(np.clip(p1, 1e-6, 1)) + b * np.log(np.clip(q, 1e-6, 1))
        pmix = np.exp(z - z.max())
        pmix /= pmix.sum()
        T = S.trifecta(pmix, P2, LAM3, P3)
        rows = []
        for bt, (names, P) in S.combos(T, pmix, P2).items():
            for k, c in enumerate(names):
                o = odds.get((bt, c))
                if o and P[0, k] * o >= 1.0:
                    rows.append({"race_id": rid, "deadline": deadline.strftime("%H:%M"), "snap": snap.strftime("%H:%M:%S"),
                                 "bet_type": bt, "combo": c, "p": round(float(P[0, k]), 5), "odds": o,
                                 "ev": round(float(P[0, k] * o), 3), "p_model_win": round(float(p1[0, int(c[0]) - 1]), 4)})
        if not rows:
            rows = [{"race_id": rid, "deadline": deadline.strftime("%H:%M"), "snap": snap.strftime("%H:%M:%S"),
                     "bet_type": "none", "combo": "", "p": 0, "odds": 0, "ev": 0, "p_model_win": 0}]
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        n = sum(r["bet_type"] != "none" for r in rows)
        print(f"{snap:%H:%M:%S} {rid} 締切{deadline:%H:%M} 候補{n}件", flush=True)
    except Exception as e:
        print("ERROR", rid, e, flush=True)


# ---------- 答え合わせ ----------
def settle(day):
    d = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:]))
    download.fetch("K", d)
    sh("tools/parse.py")
    from backtest import Book
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    book = Book(con, day, day)
    c = pd.read_csv(LIVE / f"{day}_cands.csv")
    c = c[c.bet_type != "none"]
    c["back"] = [book.settle(r, bt, cb, 100) if r in book.races else np.nan
                 for r, bt, cb in zip(c.race_id, c.bet_type, c.combo)]
    c = c.dropna(subset=["back"])
    c.to_csv(LIVE / f"{day}_settled.csv", index=False)
    print(f"{day} 記録したレース {c.race_id.nunique()}")
    for th in (1.0, 1.2, 1.5, 2.0):
        for bt, g in c[c.ev >= th].groupby("bet_type"):
            print(f"  EV>={th} {bt:<11} {len(g):4d}点  的中 {(g.back > 100).mean()*100:5.1f}%  回収率 {g.back.mean():6.1f}%")


if __name__ == "__main__":
    cmd = sys.argv[1]
    day = sys.argv[2] if len(sys.argv) > 2 else today()
    {"train": lambda: train(), "prepare": lambda: prepare(day), "run": lambda: run(day),
     "settle": lambda: settle(day)}[cmd]()
