from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from src.analysis.pipeline import AnalysisPipeline
from src.config.settings import Settings
from src.domain.models import AnalysisBundle, Article, Comment
from src.ingestion.normalized_loader import load_dataset, parse_dataset
from src.ui.views import (
    render_emotion,
    render_notice,
    render_opinion,
    render_quality,
    render_relation,
)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_article.json"


def _progress_callback(bar: Any, label: Any) -> Callable[[str, float], None]:
    def update(message: str, value: float) -> None:
        label.write(message)
        bar.progress(value)

    return update


def _manual_dataset(title: str, body: str, comments_text: str) -> tuple[Article, list[Comment]]:
    article = Article(
        article_id="manual",
        title=title or "無題の記事",
        body=body,
        source_name="手動入力",
    )
    comments = [
        Comment(
            comment_id=f"manual-{index + 1:03d}",
            article_id=article.article_id,
            text=line.strip(),
            order_index=index,
        )
        for index, line in enumerate(comments_text.splitlines())
        if line.strip()
    ]
    return article, comments


def _csv_comments(
    article_id: str, raw: bytes, fallback_text: str = ""
) -> list[Comment]:
    import csv

    rows = list(csv.DictReader(StringIO(raw.decode("utf-8-sig"))))
    comments = []
    for index, row in enumerate(rows):
        posted_at = row.get("posted_at") or None
        comments.append(
            Comment(
                comment_id=row.get("comment_id") or f"csv-{index + 1:05d}",
                article_id=article_id,
                text=row.get("text") or fallback_text,
                posted_at=datetime.fromisoformat(posted_at) if posted_at else None,
                order_index=int(row.get("order_index") or index),
                empathy_count=int(row["empathy_count"])
                if row.get("empathy_count")
                else None,
                reply_count=int(row["reply_count"]) if row.get("reply_count") else None,
                parent_comment_id=row.get("parent_comment_id") or None,
            )
        )
    return comments


def render_start(settings: Settings) -> None:
    st.header("分析開始")
    st.write(
        "許可されたAPIやユーザー提供データだけを対象にします。"
        "このアプリはYahoo!ニュースを含む外部サイトのスクレイピングを行いません。"
    )
    source = st.radio(
        "入力方法",
        ["同梱サンプル", "正規化JSON", "コメントCSV＋手動本文", "手動入力"],
        horizontal=True,
    )
    article: Article
    comments: list[Comment]
    if source == "同梱サンプル":
        article, comments = load_dataset(SAMPLE_PATH)
    elif source == "正規化JSON":
        uploaded = st.file_uploader("ArticleとCommentを含むJSON", type=["json"])
        if uploaded is None:
            st.info("JSONを選択すると分析できます。形式はREADMEを参照してください。")
            return
        try:
            article, comments = parse_dataset(json.load(uploaded))
        except (json.JSONDecodeError, ValidationError) as exc:
            st.error(f"正規化JSONを読み込めません: {exc}")
            return
    elif source == "コメントCSV＋手動本文":
        title = st.text_input("記事タイトル")
        body = st.text_area("記事本文", height=180)
        uploaded_csv = st.file_uploader(
            "コメントCSV（必須列: text）", type=["csv"]
        )
        article = Article(
            article_id="csv-upload",
            title=title or "無題の記事",
            body=body,
            source_name="CSVアップロード",
        )
        if uploaded_csv is None:
            st.info("CSVを選択すると分析できます。")
            return
        try:
            comments = _csv_comments(article.article_id, uploaded_csv.getvalue())
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(f"CSVを読み込めません: {exc}")
            return
    else:
        title = st.text_input("記事タイトル")
        body = st.text_area("記事本文", height=180)
        comments_text = st.text_area("コメント（1行1件）", height=180)
        article, comments = _manual_dataset(title, body, comments_text)

    mode = st.selectbox(
        "分析モード",
        ["quick", "standard", "full"],
        index=["quick", "standard", "full"].index(
            settings.analysis_default_mode
            if settings.analysis_default_mode in {"quick", "standard", "full"}
            else "standard"
        ),
        format_func=lambda value: {
            "quick": "クイック（最大500件）",
            "standard": "標準（設定上限）",
            "full": "フル（設定上限）",
        }[value],
    )
    mode_limit = {
        "quick": 500,
        "standard": settings.analysis_max_comments,
        "full": settings.analysis_max_comments,
    }[mode]
    comments = comments[:mode_limit]
    left, right, third = st.columns(3)
    left.metric("コメント件数", len(comments))
    right.metric(
        "推定LLMバッチ数",
        (len(comments) + settings.comment_batch_size - 1)
        // settings.comment_batch_size,
    )
    third.metric("APIモード", "OpenAI" if settings.openai_api_key else "ローカル暫定")
    st.write(f"**{article.title}**")
    with st.expander("本文を確認"):
        st.write(article.body or "本文なし")
    analyze, rerun = st.columns(2)
    start = analyze.button("分析開始", type="primary", use_container_width=True)
    force = rerun.button("再分析（キャッシュを更新）", use_container_width=True)
    if start or force:
        bar = st.progress(0.0)
        label = st.empty()
        try:
            pipeline = AnalysisPipeline(settings)
            bundle = pipeline.run(
                article,
                comments,
                force=force,
                progress=_progress_callback(bar, label),
            )
            st.session_state["bundle_json"] = bundle.model_dump_json()
            st.success(
                "分析が完了しました。"
                + ("（キャッシュを再利用）" if bundle.cache_hit else "")
                + " 上部の各タブで結果を確認できます。"
            )
            st.rerun()
        except Exception as exc:
            st.error(
                f"分析を完了できませんでした: {type(exc).__name__}: {exc}"
            )
            st.info("入力形式と環境変数を確認してください。API障害時は再実行できます。")


def main() -> None:
    st.set_page_config(
        page_title="ニュース・コメント分析",
        page_icon="🛰️",
        layout="wide",
    )
    st.title("ニュース・コメント分析ラボ")
    settings = Settings.from_env()
    start_tab, relation_tab, opinion_tab, emotion_tab, quality_tab = st.tabs(
        ["分析開始", "本文との関係", "意見空間", "感情・拡散", "議論品質"]
    )
    with start_tab:
        render_start(settings)
    bundle_json = st.session_state.get("bundle_json")
    if bundle_json:
        bundle = AnalysisBundle.model_validate_json(bundle_json)
        for warning in bundle.warnings:
            st.sidebar.warning(warning)
        st.sidebar.caption(
            f"分析ID {bundle.run.run_id[:8]} / {bundle.run.comment_count}件 / "
            f"{'cache' if bundle.cache_hit else 'fresh'}"
        )
        with relation_tab:
            render_relation(bundle)
        with opinion_tab:
            render_opinion(bundle)
        with emotion_tab:
            render_emotion(bundle)
        with quality_tab:
            render_quality(bundle)
    else:
        for tab in (relation_tab, opinion_tab, emotion_tab, quality_tab):
            with tab:
                st.info("「分析開始」タブでデータを選び、分析を実行してください。")
    render_notice()


if __name__ == "__main__":
    main()
