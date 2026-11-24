"""締切前のオッズで「買ったつもり」を記録する紙上検証（お金は賭けない）。
確定オッズでの検証は、締切後にしか分からないオッズで選んでいる偏りがある。
実戦で見える「締切数分前のオッズ」でも期待値の高い舟券が当たるのかを、ここで確かめる。

使い方:
  python3 tools/live.py prepare [YYYYMMDD]         # 当日の番組表を取得し、当日分の特徴量とレース一覧を作る
  python3 tools/live.py train [YYYYMMDD]           # その日の前日までの全データで学習し data/model/ に保存
  python3 tools/live.py run [YYYYMMDD] [開始 終了]  # 取得時刻（締切5分前）が 開始〜終了(HH:MM) のレースを記録する
  python3 tools/live.py settle [YYYYMMDD]          # 結果が出ている未照合の日をすべて答え合わせし、集計を作り直す

毎日の実行は GitHub Actions（.github/workflows/live.yml）。1回6時間までなので、1日を朝・昼・夕の3回に分ける。

記録（リポジトリにコミットする）:
  records/live/<日付>_cands_<区分>.csv … 期待値1以上の全候補（券種・組・確率・締切前オッズ・取得時刻）
  records/live/<日付>_settled.csv      … 候補ごとの払戻（1点100円）
  records/live/SUMMARY.md              … 全期間の集計（日単位ブートストラップの95%区間つき）
取得したページそのものは data/raw/live/<日付>/ に保存（Actions では成果物として90日残す）

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
LIVE = ROOT / "live"                       # その日の特徴量・レース一覧（作業用）
MODEL = ROOT / "model"                     # 学習済みモデル
REC = HERE.parent / "records" / "live"     # 記録（コミットする）
FULL = "features_live.parquet"             # prepare が作る全期間の特徴量（train が使う）
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
def train(day):
    MODEL.mkdir(parents=True, exist_ok=True)
    src = ROOT / FULL if (ROOT / FULL).exists() else ROOT / "features.parquet"
    df = pd.read_parquet(src)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df.date < f"{day[:4]}-{day[4:6]}-{day[6:]}"]   # 当日以降（結果なし）は使わない
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
    m1.save_model(str(MODEL / "m1.txt"), num_iteration=m1.best_iteration)
    m2.save_model(str(MODEL / "m2.txt"), num_iteration=m2.best_iteration)
    m3.save_model(str(MODEL / "m3.txt"), num_iteration=m3.best_iteration)
    cfg = {"trained_through": last, "t1": float(t1), "t2": t2, "t3": t3, "f2": f2, "f3": f3, "feats": FEATS,
           "cats": {c: list(df[c].cat.categories) for c in cats if hasattr(df[c], "cat")}}
    (MODEL / "config.json").write_text(json.dumps(cfg, ensure_ascii=False))
    print("学習完了", last, "温度", t1, t2, t3)


# ---------- 準備 ----------
def prepare(day):
    d = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:]))
    for k in ("K", "B"):
        download.fetch(k, d - datetime.timedelta(days=1))
    download.fetch("B", d)
    sh("tools/parse.py")
    sh("tools/features.py", "--keep-open", day, "--out", FULL)
    LIVE.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(ROOT / FULL)
    f = f[f["date"].dt.strftime("%Y%m%d") == day]
    f.to_parquet(LIVE / f"feats_{day}.parquet")
    races = pd.read_sql("SELECT race_id, venue, rno, deadline FROM races WHERE date=? AND deadline IS NOT NULL",
                        sqlite3.connect(ROOT / "boatrace.sqlite"), params=(day,))
    races.to_csv(LIVE / f"races_{day}.csv", index=False)
    print("当日のレース", f.race_id.nunique(), "締切あり", len(races))


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


def hm(day, s):
    hh, mm = map(int, s.split(":"))
    base = datetime.datetime(int(day[:4]), int(day[4:6]), int(day[6:]), tzinfo=JST)
    return base + datetime.timedelta(hours=hh, minutes=mm)   # "24:00" も可


def run(day, start="00:00", end="24:00", part="all"):
    """取得時刻（締切 LEAD_MIN 分前）が [start, end) に入るレースだけを受け持つ。
    朝・昼・夕の3回に分けて動かすので、受け持ちが重ならないよう区切る。"""
    cfg = json.loads((MODEL / "config.json").read_text())
    m1, m2, m3 = (lgb.Booster(model_file=str(MODEL / f"m{k}.txt")) for k in (1, 2, 3))
    feats = pd.read_parquet(LIVE / f"feats_{day}.parquet")
    for c, cats in cfg["cats"].items():
        feats[c] = pd.Categorical(feats[c], categories=cats)
    rc = LIVE / f"races_{day}.csv"
    if rc.exists():
        races = pd.read_csv(rc).itertuples(index=False, name=None)
    else:
        races = sqlite3.connect(ROOT / "boatrace.sqlite").execute(
            "SELECT race_id, venue, rno, deadline FROM races WHERE date=? AND deadline IS NOT NULL", (day,)).fetchall()
    lo, hi = hm(day, start), hm(day, end)
    lead = datetime.timedelta(minutes=LEAD_MIN)
    todo = {}
    for rid, v, r, dl in races:
        t = hm(day, dl)
        if lo <= t - lead < hi:
            todo[rid] = (int(v), int(r), t)
    REC.mkdir(parents=True, exist_ok=True)
    out = REC / f"{day}_cands_{part}.csv"
    done = set()
    for f in REC.glob(f"{day}_cands*.csv"):
        done |= set(pd.read_csv(f).race_id)
    print(f"{day} {part} {start}-{end}: 受け持ち {len(todo)}レース（記録済み {len(done & set(todo))}）", flush=True)
    pool = ThreadPoolExecutor(8)
    while todo:
        now = datetime.datetime.now(JST)
        due = [rid for rid, (v, r, t) in todo.items()
               if rid not in done and t - lead <= now < t - datetime.timedelta(minutes=1)]
        for rid in due:
            done.add(rid)
            pool.submit(record, day, rid, *todo[rid], feats, cfg, m1, m2, m3, out)
        if all(t - datetime.timedelta(minutes=1) <= now for _, _, t in todo.values()):
            break
        time.sleep(15)
    pool.shutdown(wait=True)
    n = len(pd.read_csv(out).race_id.unique()) if out.exists() else 0
    print(f"{day} {part}: 記録 {n}レース / 受け持ち {len(todo)}", flush=True)


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
THS = (1.0, 1.2, 1.5, 2.0)


def settle(today_):
    """結果が DB に入っていて、まだ照合していない日をすべて照合し、全期間の集計を作り直す。
    結果ファイル(K)は翌朝に出るので、DB は prepare（前日分のKを取って作り直す）の後のものを使う。"""
    from backtest import Book
    con = sqlite3.connect(ROOT / "boatrace.sqlite")
    days = sorted({f.name[:8] for f in REC.glob("*_cands*.csv")})
    for day in days:
        if day >= today_ or (REC / f"{day}_settled.csv").exists():
            continue
        book = Book(con, day, day)
        if not book.races:
            print(day, "結果がまだ無い")
            continue
        c = pd.concat([pd.read_csv(f) for f in sorted(REC.glob(f"{day}_cands*.csv"))])
        c = c[c.bet_type != "none"].drop_duplicates(["race_id", "bet_type", "combo"])
        c["back"] = [book.settle(r, bt, cb, 100) if r in book.races else np.nan
                     for r, bt, cb in zip(c.race_id, c.bet_type, c.combo)]
        c = c.dropna(subset=["back"])
        c.to_csv(REC / f"{day}_settled.csv", index=False)
        print(day, "照合", c.race_id.nunique(), "レース")
    summary()


def summary():
    fs = sorted(REC.glob("*_settled.csv"))
    if not fs:
        return
    c = pd.concat([pd.read_csv(f).assign(day=f.name[:8]) for f in fs])
    rng = np.random.default_rng(0)
    rows = []
    for th in THS:
        for bt, g in c[c.ev >= th].groupby("bet_type"):
            by = g.groupby("day").agg(inv=("back", "size"), back=("back", "sum"))
            inv, back = by.inv.to_numpy() * 100, by.back.to_numpy()
            idx = rng.integers(0, len(by), (2000, len(by)))
            boot = back[idx].sum(1) / inv[idx].sum(1) * 100
            wo = (g.back.sum() - g.back.max()) / (len(g) * 100) * 100
            rows.append((bt, th, len(by), g.race_id.nunique(), len(g), (g.back > 0).mean() * 100,
                         back.sum() / inv.sum() * 100, np.percentile(boot, 2.5), np.percentile(boot, 97.5), wo))
    t = pd.DataFrame(rows, columns=["券種", "EV≥", "日数", "レース", "点数", "払戻あり%", "回収率%", "下限%", "上限%", "最大除く%"])
    order = {b: i for i, b in enumerate(["tan", "fuku", "wide", "nirenpuku", "nirentan", "sanrenpuku", "sanrentan"])}
    t = t.sort_values(["券種", "EV≥"], key=lambda x: x.map(order) if x.name == "券種" else x)
    t.to_csv(REC / "summary.csv", index=False)
    lines = ["# 締切前オッズでの紙上記録（お金は賭けていない）", "",
             f"期間 {fs[0].name[:8]}〜{fs[-1].name[:8]}（{len(fs)}日）。1点100円で、期待値（混合確率×締切5分前オッズ）が閾値以上の舟券をすべて買ったとき。",
             "95%区間は日単位のブートストラップ。**下限が100%を超え続けて初めて『利益が出る』と言える。**", "",
             "| 券種 | EV≥ | 日数 | レース | 点数 | 払戻あり | 回収率 | 95%区間 | 最大払戻を除く |", "|---|---|---|---|---|---|---|---|---|"]
    jp = {"tan": "単勝", "fuku": "複勝", "wide": "拡連複", "nirenpuku": "2連複", "nirentan": "2連単",
          "sanrenpuku": "3連複", "sanrentan": "3連単"}
    for r in t.itertuples(index=False):
        lines.append(f"| {jp.get(r[0], r[0])} | {r[1]:.1f} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f}% | {r[6]:.1f}% | {r[7]:.0f}〜{r[8]:.0f}% | {r[9]:.1f}% |")
    lines += ["", "日別の明細は `<日付>_settled.csv`。自動更新（.github/workflows/live.yml）。"]
    (REC / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    day = sys.argv[2] if len(sys.argv) > 2 else today()
    {"train": lambda: train(day), "prepare": lambda: prepare(day),
     "run": lambda: run(day, *sys.argv[3:6]),
     "settle": lambda: settle(day)}[cmd]()
