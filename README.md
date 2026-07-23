<div align="center">

# Comment Analysis

**ニュースとコメントの「あいだ」を可視化する。**

記事本文と正規化済みコメントを、感情・論点・意見空間・議論品質から読み解く  
Streamlit製の分析ラボです。

</div>

---

### Highlights

🪐 Opinion Galaxy　　🌋 感情地震計　　🔥 本文トリガー  
🧭 認識ギャップ　　🌱 少数意見　　　🕸️ 言い換え拡散  
📐 議論品質　　　　🌊 論点脱線　　　📰 見出し分析

すべての結果から、根拠となる本文または元コメントを確認できます。

### Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
streamlit run app.py
```

同梱サンプル、正規化JSON、CSV、手動入力に対応しています。

### OpenAI

```dotenv
OPENAI_API_KEY=
OPENAI_TEXT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

モデルIDや分析閾値は [.env.example](.env.example) で変更できます。API未設定時は、
決定的なローカル分析へ自動で切り替わります。

> 外部サイトのスクレイピングは行いません。分析結果は取得データに基づく推定であり、
> 社会全体の世論、因果関係、投稿者の属性や人格を判定するものではありません。
