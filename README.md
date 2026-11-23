# boatrace-yosou

ボートレースの買い方ルールを、BOAT RACE公式の過去データ（数万レース）で検証するための道具。

```
pip install -r requirements.txt
python3 tools/download.py 2023-01-01 2026-11-22   # 公式データ取得（data/raw）
python3 tools/parse.py                            # data/boatrace.sqlite を作る
python3 tools/backtest.py                         # 基準となる買い方の回収率
python3 tools/fetch_odds.py 2025-07-01 2026-11-22 12  # 確定オッズ（12日おき）
python3 tools/parse_odds.py                       # オッズをDBへ
python3 tools/features.py                         # 特徴量（前日までの情報だけ）
python3 tools/model.py [--ex]                     # 1着確率モデル（--ex: 展示タイムも使う）
python3 tools/bet.py [base|ex]                    # オッズと比べた買い方の評価
```

検証の方針は `CLAUDE.md` を参照。
