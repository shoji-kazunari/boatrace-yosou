# boatrace-yosou

ボートレースの買い方ルールを、BOAT RACE公式の過去データ（数万レース）で検証するための道具。

```
pip install -r requirements.txt
python3 tools/download.py 2023-01-01 2026-11-22   # 公式データ取得（data/raw）
python3 tools/parse.py                            # data/boatrace.sqlite を作る
python3 tools/backtest.py                         # 基準となる買い方の回収率
```

検証の方針は `CLAUDE.md` を参照。
