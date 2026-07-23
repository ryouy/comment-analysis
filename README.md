<div align="center">

# Comment Analysis

### ニュースとコメントの「あいだ」を可視化する。

感情、論点、少数意見、拡散、議論品質を探索するStreamlitアプリ。

`Opinion Galaxy` · `感情地震計` · `認識ギャップ` · `論点脱線` · `見出し分析`

</div>

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
streamlit run app.py
```

同梱サンプル、JSON、CSV、手動入力に対応。OpenAIの設定は
[`.env.example`](.env.example) を参照してください。API未設定でも動作します。

> 外部サイトをスクレイピングしません。分析結果は推定であり、世論・因果・人物を判定するものではありません。
