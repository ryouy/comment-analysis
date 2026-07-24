from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.domain.models import AnalysisBundle

EMOTION_LABELS = {
    "anger": "怒り",
    "anxiety": "不安",
    "disappointment": "失望",
    "ridicule": "嘲笑",
    "empathy": "共感",
    "hope": "希望",
    "doubt": "疑念",
    "resignation": "諦め",
    "moral_outrage": "道徳的憤り",
}
JST = ZoneInfo("Asia/Tokyo")

DIMENSION_LABELS = {
    "specificity": "具体性",
    "evidence": "根拠",
    "originality": "独自性",
    "logical_coherence": "論理性",
    "relevance": "記事との関連",
    "constructiveness": "建設性",
    "respectfulness": "敬意",
    "information_density": "情報量",
}


def _comment_map(bundle: AnalysisBundle) -> dict[str, str]:
    return {comment.comment_id: comment.text for comment in bundle.comments}


def _format_posted_at(value: datetime | str | None) -> str:
    if value is None:
        return "投稿時刻なし"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    localized = value.replace(tzinfo=JST) if value.tzinfo is None else value.astimezone(JST)
    return localized.strftime("%Y/%m/%d %H:%M")


def evidence_comments(
    bundle: AnalysisBundle,
    comment_ids: list[str],
    key: str,
    label: str = "関連コメント",
) -> None:
    available = [item for item in comment_ids if item in _comment_map(bundle)]
    if not available:
        st.caption(f"{label}: 該当なし")
        return
    comments_by_id = {comment.comment_id: comment for comment in bundle.comments}
    selected = st.selectbox(
        label,
        available,
        key=key,
        format_func=lambda comment_id: (
            f"{_format_posted_at(comments_by_id[comment_id].posted_at)} ｜ "
            f"{comments_by_id[comment_id].text[:64]}"
        ),
    )
    comment = next(item for item in bundle.comments if item.comment_id == selected)
    analysis = next(item for item in bundle.analyses if item.comment_id == selected)
    with st.container(border=True):
        st.write(comment.text)
        st.caption(
            f"{_format_posted_at(comment.posted_at)} ・ 投稿順 {comment.order_index + 1} ・ "
            f"主な感情: {EMOTION_LABELS.get(analysis.dominant_emotion or '', '不明')}"
        )
        if analysis.article_sentence_links:
            sentence_id = st.selectbox(
                "対応する記事本文",
                [link.sentence_id for link in analysis.article_sentence_links],
                key=f"{key}-sentence",
            )
            sentence = next(item for item in bundle.sentences if item.sentence_id == sentence_id)
            st.info(sentence.text)


def render_relation(bundle: AnalysisBundle) -> None:
    st.header("記事との関連")
    heatmap = bundle.features["trigger_heatmap"]
    rows = pd.DataFrame(heatmap["sentences"])
    st.subheader("本文への反応")
    mode = st.selectbox(
        "表示する指標",
        ["comment_count", "anger", "anxiety", "empathy"],
        format_func=lambda value: {
            "comment_count": "反応量",
            "anger": "怒り",
            "anxiety": "不安",
            "empathy": "支持・共感",
        }[value],
    )
    if mode != "comment_count":
        rows["表示値"] = rows["emotions"].map(lambda value: value.get(mode, 0))
    else:
        rows["表示値"] = rows["comment_count"]
    st.dataframe(
        rows[["sentence_id", "text", "is_headline", "comment_count", "表示値"]],
        width="stretch",
        hide_index=True,
    )
    sentence_id = st.selectbox(
        "記事本文を選ぶ",
        rows["sentence_id"],
        format_func=lambda value: next(
            item.text for item in bundle.sentences if item.sentence_id == value
        )[:80],
    )
    row = next(item for item in heatmap["sentences"] if item["sentence_id"] == sentence_id)
    evidence_comments(bundle, row["comment_ids"], "heatmap-comments")
    if heatmap["outside_comment_ids"]:
        with st.expander("記事本文に直接対応しないコメント"):
            evidence_comments(
                bundle, heatmap["outside_comment_ids"], "outside-comments"
            )

    st.subheader("記事とコメントの認識差")
    gap = bundle.features["gap_index"]
    st.metric("認識差", f"{gap['value']:.1f} / 100")
    st.caption(f"信頼度 {gap['confidence'] * 100:.0f}%")
    if gap["warning"]:
        st.warning(gap["warning"])
    component_frame = pd.DataFrame(
        [
            {"構成要素": key, "値": value * 100, "重み": gap["weights"][key]}
            for key, value in gap["components"].items()
        ]
    )
    st.plotly_chart(
        px.bar(component_frame, x="値", y="構成要素", orientation="h"),
        width="stretch",
    )
    with st.expander("計算方法"):
        st.code(
            "100 × (0.30×意味的不一致 + 0.25×本文外率 + 0.20×見出し依存率"
            " + 0.15×解釈衝突率 + 0.10×本文カバレッジ不足)"
        )

    st.subheader("論点の広がり")
    drift = bundle.features["topic_drift"]
    if drift["links"]:
        labels = {node["id"]: node["label"] for node in drift["nodes"]}
        cluster_ids = sorted({link["target"] for link in drift["links"]})
        all_ids = list(labels) + cluster_ids
        fig = go.Figure(
            go.Sankey(
                node={"label": [labels.get(item, item) for item in all_ids]},
                link={
                    "source": [all_ids.index(link["source"]) for link in drift["links"]],
                    "target": [all_ids.index(link["target"]) for link in drift["links"]],
                    "value": [link["value"] for link in drift["links"]],
                    "customdata": [link["category"] for link in drift["links"]],
                    "hovertemplate": "%{customdata}: %{value}件<extra></extra>",
                },
            )
        )
        st.plotly_chart(fig, width="stretch")
        link_index = st.selectbox(
            "関連を選ぶ",
            range(len(drift["links"])),
            format_func=lambda index: (
                f"{drift['links'][index]['source']} → "
                f"{drift['links'][index]['target']} "
                f"({drift['links'][index]['category']})"
            ),
        )
        evidence_comments(
            bundle, drift["links"][link_index]["comment_ids"], "drift-comments"
        )

    st.subheader("見出しの比較")
    framing = bundle.features["framing"]
    st.write("元の見出し:", framing["original_title"])
    st.write("記事本文との一致:", f"{framing['body_alignment'] * 100:.0f}%")
    st.write("強調表現:", framing["emotional_terms"] + framing["emphasis_terms"] or "なし")
    st.caption("記事本文を基にした比較案です。正解を示すものではありません。")
    headline_labels = {
        "fact_focused": "事実中心",
        "context_focused": "背景を補足",
        "cautious": "慎重な表現",
    }
    st.table(
        pd.DataFrame(
            [
                {"方針": headline_labels.get(key, key), "見出し": value}
                for key, value in framing["neutral_headlines"].items()
            ]
        )
    )
    evidence_comments(
        bundle,
        framing["dependent_comment_ids"],
        "framing-dependent",
        "見出しの影響が見られるコメント",
    )
    st.caption(framing["note"])


def render_opinion(bundle: AnalysisBundle) -> None:
    st.header("意見分布")
    galaxy = bundle.features["galaxy"]
    st.subheader("コメントの分布")
    if galaxy["available"]:
        frame = pd.DataFrame(galaxy["points"])
        fig = px.scatter(
            frame,
            x="x",
            y="y",
            color="cluster_label",
            size=frame["empathy_count"].map(lambda value: max(5, value + 1)),
            symbol="is_minority",
            hover_data=["comment_id", "dominant_emotion", "originality", "relevance"],
            render_mode="webgl",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{galaxy['reducer']} / {galaxy['clusterer']}（同一入力で再現可能）")
        evidence_comments(
            bundle, frame["comment_id"].tolist(), "galaxy-comment", "コメントを選ぶ"
        )
    else:
        st.info(galaxy["reason"])
    st.dataframe(
        pd.DataFrame([item.model_dump(exclude={"dominant_emotions"}) for item in bundle.clusters]),
        width="stretch",
    )

    st.subheader("少数意見")
    minority = bundle.features["minority_signals"]
    if minority:
        st.dataframe(pd.DataFrame(minority), width="stretch", hide_index=True)
        evidence_comments(
            bundle,
            [item["comment_id"] for item in minority],
            "minority-comment",
            "少数意見の根拠",
        )
    else:
        st.info("該当する少数意見はありません。")

    st.subheader("特徴的なフレーズ")
    phrases = bundle.features["semantic_cloud"]
    if phrases:
        phrase_frame = pd.DataFrame(phrases)
        st.plotly_chart(
            px.bar(
                phrase_frame.head(20),
                x="score",
                y="phrase",
                orientation="h",
                color="dominant_emotion",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.scatter(
                phrase_frame,
                x="score",
                y="phrase",
                size="count",
                color="dominant_emotion",
            ),
            width="stretch",
        )
        selected_phrase = st.selectbox("フレーズを選択", [item["phrase"] for item in phrases])
        selected = next(item for item in phrases if item["phrase"] == selected_phrase)
        evidence_comments(bundle, selected["comment_ids"], "phrase-comments")


def render_emotion(bundle: AnalysisBundle) -> None:
    st.header("感情の推移")
    timeline = bundle.features["emotion_timeline"]
    st.subheader("感情の推移")
    st.info(
        "投稿時刻を基準に表示しています。"
        if timeline["axis_mode"] == "time"
        else "投稿時刻が不足しているため、投稿順で表示しています。"
    )
    if timeline["bins"]:
        emotions = st.multiselect(
            "表示する感情",
            list(EMOTION_LABELS),
            default=["anger", "anxiety", "empathy", "hope", "moral_outrage"],
            format_func=lambda name: EMOTION_LABELS[name],
        )
        weighted = st.toggle("共感数を反映", value=False)
        smoothed = st.toggle("推移をなめらかに表示", value=True)
        key = "weighted_mean" if weighted else "mean"
        x_axis_label = (
            "投稿時刻" if timeline["axis_mode"] == "time" else "投稿順"
        )
        long_rows = [
            {
                x_axis_label: (
                    item["timestamp"]
                    if timeline["axis_mode"] == "time"
                    else item["end_order"] + 1
                ),
                "感情": EMOTION_LABELS[emotion],
                "強度": item[key][emotion],
                "コメント数": item["comment_count"],
            }
            for item in timeline["bins"]
            for emotion in emotions
        ]
        timeline_frame = pd.DataFrame(long_rows)
        if smoothed and not timeline_frame.empty:
            timeline_frame["強度"] = timeline_frame.groupby("感情")["強度"].transform(
                lambda series: series.ewm(span=3, adjust=False).mean()
            )
        fig = px.line(
            timeline_frame,
            x=x_axis_label,
            y="強度",
            color="感情",
            markers=True,
        )
        for point in timeline["change_points"]:
            marker_x = (
                timeline["bins"][point["position"]]["timestamp"]
                if timeline["axis_mode"] == "time"
                else timeline["bins"][point["position"]]["end_order"] + 1
            )
            fig.add_vline(x=marker_x, line_dash="dash", line_color="#d62728")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"集計単位: {timeline['window_size']}コメント / "
            f"変化が大きい箇所: {len(timeline['change_points'])}件"
        )
        if timeline["change_points"]:
            cp_index = st.selectbox(
                "変化が大きい箇所",
                range(len(timeline["change_points"])),
                format_func=lambda index: (
                    f"{_format_posted_at(timeline['change_points'][index]['timestamp'])} / "
                    f"変化量 {timeline['change_points'][index]['magnitude']:.2f}"
                ),
            )
            point = timeline["change_points"][cp_index]
            st.write(
                "影響感情:",
                [
                    EMOTION_LABELS.get(item, item)
                    for item in point["affected_emotions"]
                ],
            )
            st.write("同時期に増えた候補:", point["candidate_triggers"])
            comparison = pd.DataFrame(
                {
                    "感情": [EMOTION_LABELS[item] for item in point["before_vector"]],
                    "前": list(point["before_vector"].values()),
                    "後": list(point["after_vector"].values()),
                }
            )
            st.dataframe(comparison, hide_index=True)
            evidence_comments(
                bundle,
                point["representative_comment_ids"],
                "change-point-comments",
                "この付近のコメント",
            )
        else:
            st.info("大きな変化は見つかりませんでした。")
        st.caption(timeline["note"])

    st.subheader("類似コメントの広がり")
    propagation = bundle.features["propagation"]
    st.metric("完全重複率", f"{propagation['duplicate_rate'] * 100:.1f}%")
    st.metric("言い換え率", f"{propagation['paraphrase_rate'] * 100:.1f}%")
    if propagation["edges"]:
        edge_frame = pd.DataFrame(propagation["edges"])
        node_frame = pd.DataFrame(propagation["nodes"])
        cluster_names = {
            name: index
            for index, name in enumerate(
                sorted(node_frame["cluster_id"].fillna("未分類").unique())
            )
        }
        node_frame["group_y"] = node_frame["cluster_id"].fillna("未分類").map(cluster_names)
        node_lookup = node_frame.set_index("comment_id")
        graph = go.Figure()
        for edge in propagation["edges"][:500]:
            source = node_lookup.loc[edge["source"]]
            target = node_lookup.loc[edge["target"]]
            graph.add_trace(
                go.Scatter(
                    x=[source["order_index"], target["order_index"]],
                    y=[source["group_y"], target["group_y"]],
                    mode="lines",
                    line={"width": 1 + edge["similarity"] * 2},
                    hoverinfo="text",
                    text=edge["kind"],
                    showlegend=False,
                )
            )
        graph.add_trace(
            go.Scatter(
                x=node_frame["order_index"],
                y=node_frame["group_y"],
                mode="markers",
                marker={
                    "size": node_frame["empathy_count"].map(
                        lambda value: min(28, 7 + value**0.5)
                    )
                },
                text=node_frame["comment_id"],
                name="コメント",
            )
        )
        graph.update_layout(xaxis_title="投稿順", yaxis_title="類似グループ")
        st.plotly_chart(graph, width="stretch")
        st.dataframe(edge_frame, width="stretch", hide_index=True)
        edge_index = st.selectbox(
            "類似関係",
            range(len(propagation["edges"])),
            format_func=lambda index: (
                f"{propagation['edges'][index]['source']} → "
                f"{propagation['edges'][index]['target']} "
                f"({propagation['edges'][index]['kind']})"
            ),
        )
        edge = propagation["edges"][edge_index]
        evidence_comments(
            bundle, [edge["source"], edge["target"]], "propagation-comments"
        )
    st.caption(propagation["note"])


def render_quality(bundle: AnalysisBundle) -> None:
    st.header("コメント品質")
    st.subheader("全体指標")
    health = bundle.features["health"]
    health_frame = pd.DataFrame(
        [{"指標": key, "値": value} for key, value in health["metrics"].items()]
    )
    st.plotly_chart(
        px.bar(health_frame, x="値", y="指標", orientation="h", range_x=[0, 100]),
        width="stretch",
    )
    st.caption(f"信頼度 {health['confidence'] * 100:.0f}% — {health['warning']}")
    with st.expander("指標定義"):
        st.write(
            "多様性=クラスター分布、分断度=クラスター分離の推定、本文関連性=本文との意味類似、"
            "根拠提示率=理由・出典表現、建設性=提案表現、重複度=完全一致と言い換え、"
            "敬意=攻撃表現の少なさ、少数意見可視性=小規模意見群の比率、橋渡し率=中間密度です。"
        )

    st.subheader("コメントごとの品質")
    quality = pd.DataFrame(bundle.features["quality_frontier"])
    dimensions = [
        "specificity",
        "evidence",
        "originality",
        "logical_coherence",
        "relevance",
        "constructiveness",
        "respectfulness",
        "information_density",
    ]
    left, right = st.columns(2)
    x_axis = left.selectbox(
        "横軸", dimensions, index=1, format_func=lambda value: DIMENSION_LABELS[value]
    )
    y_axis = right.selectbox(
        "縦軸", dimensions, index=2, format_func=lambda value: DIMENSION_LABELS[value]
    )
    if not quality.empty:
        st.plotly_chart(
            px.scatter(
                quality,
                x=x_axis,
                y=y_axis,
                color="cluster_id",
                size=quality["empathy_count"].map(lambda value: max(5, value + 1)),
                hover_data=["comment_id", "insufficient_material"],
            ),
            width="stretch",
        )
        evidence_comments(
            bundle, quality["comment_id"].tolist(), "quality-comments", "コメントを選ぶ"
        )
    st.caption("共感数は点サイズにのみ使用し、品質スコアとは区別しています。")

    st.subheader("表現上の特徴")
    rhetoric = bundle.features["rhetoric_lens"]
    threshold = st.slider("表示確率の下限", 0.60, 1.00, 0.60, 0.05)
    visible = [
        item for item in rhetoric["items"] if item["probability"] >= threshold
    ]
    if visible:
        st.dataframe(pd.DataFrame(visible), width="stretch", hide_index=True)
        evidence_comments(
            bundle,
            [item["comment_id"] for item in visible],
            "rhetoric-comments",
            "検出候補の元コメント",
        )
    else:
        st.info("閾値以上の候補はありません。")
    st.caption("ラベルは文章表現の候補であり、投稿者の人格・属性や真偽を判定しません。")


def render_notice() -> None:
    st.divider()
    st.caption(
        "この分析は取得できたコメントのみを対象とし、社会全体の世論を表しません。"
        "感情・立場・レトリックはAIまたはローカル特徴量による推定で、誤判定を含みます。"
        "少数意見は重要性を保証せず、類似線はコピーや影響関係を断定しません。"
        "見出し分析は編集意図を断定せず、健全性指標は人や集団の価値を評価しません。"
    )
