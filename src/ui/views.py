from __future__ import annotations

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


def _comment_map(bundle: AnalysisBundle) -> dict[str, str]:
    return {comment.comment_id: comment.text for comment in bundle.comments}


def evidence_comments(
    bundle: AnalysisBundle, comment_ids: list[str], key: str, label: str = "根拠コメント"
) -> None:
    available = [item for item in comment_ids if item in _comment_map(bundle)]
    if not available:
        st.caption(f"{label}: 該当なし")
        return
    selected = st.selectbox(label, available, key=key)
    comment = next(item for item in bundle.comments if item.comment_id == selected)
    analysis = next(item for item in bundle.analyses if item.comment_id == selected)
    with st.container(border=True):
        st.write(comment.text)
        st.caption(
            f"匿名コメント {selected} / 投稿順 {comment.order_index} / "
            f"支配感情 {EMOTION_LABELS.get(analysis.dominant_emotion or '', '不明')}"
        )
        if analysis.article_sentence_links:
            sentence_id = st.selectbox(
                "関連本文へ",
                [link.sentence_id for link in analysis.article_sentence_links],
                key=f"{key}-sentence",
            )
            sentence = next(item for item in bundle.sentences if item.sentence_id == sentence_id)
            st.info(sentence.text)


def render_relation(bundle: AnalysisBundle) -> None:
    st.header("本文との関係")
    heatmap = bundle.features["trigger_heatmap"]
    rows = pd.DataFrame(heatmap["sentences"])
    st.subheader("F02 本文トリガーヒートマップ")
    mode = st.selectbox(
        "表示モード",
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
        use_container_width=True,
        hide_index=True,
    )
    sentence_id = st.selectbox(
        "本文文を選択して関連コメントを表示",
        rows["sentence_id"],
        format_func=lambda value: next(
            item.text for item in bundle.sentences if item.sentence_id == value
        )[:80],
    )
    row = next(item for item in heatmap["sentences"] if item["sentence_id"] == sentence_id)
    evidence_comments(bundle, row["comment_ids"], "heatmap-comments")
    if heatmap["outside_comment_ids"]:
        with st.expander("本文外論点"):
            evidence_comments(
                bundle, heatmap["outside_comment_ids"], "outside-comments"
            )

    st.subheader("F05 認識ギャップ指数")
    gap = bundle.features["gap_index"]
    st.metric("認識ギャップ指数", f"{gap['value']:.1f} / 100")
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
        use_container_width=True,
    )
    with st.expander("算出式と定義"):
        st.code(
            "100 × (0.30×意味的不一致 + 0.25×本文外率 + 0.20×見出し依存率"
            " + 0.15×解釈衝突率 + 0.10×本文カバレッジ不足)"
        )

    st.subheader("F12 論点脱線サンキー図")
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
        st.plotly_chart(fig, use_container_width=True)
        link_index = st.selectbox(
            "流れを選択",
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

    st.subheader("F15 見出しフレーミング研究室")
    framing = bundle.features["framing"]
    st.write("元見出し:", framing["original_title"])
    st.write("本文一致度:", f"{framing['body_alignment'] * 100:.0f}%")
    st.write("感情・強調語:", framing["emotional_terms"] + framing["emphasis_terms"] or "なし")
    st.caption("AIによる比較用見出し（元見出しより正しいとは限りません）")
    st.table(
        pd.DataFrame(
            [{"型": key, "見出し": value} for key, value in framing["neutral_headlines"].items()]
        )
    )
    evidence_comments(
        bundle,
        framing["dependent_comment_ids"],
        "framing-dependent",
        "見出し依存推定の根拠コメント",
    )
    st.caption(framing["note"])


def render_opinion(bundle: AnalysisBundle) -> None:
    st.header("意見空間")
    galaxy = bundle.features["galaxy"]
    st.subheader("F01 Opinion Galaxy")
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{galaxy['reducer']} / {galaxy['clusterer']}（同一入力で再現可能）")
        evidence_comments(
            bundle, frame["comment_id"].tolist(), "galaxy-comment", "点を選択"
        )
    else:
        st.info(galaxy["reason"])
    st.dataframe(
        pd.DataFrame([item.model_dump(exclude={"dominant_emotions"}) for item in bundle.clusters]),
        use_container_width=True,
    )

    st.subheader("F07 少数意見発見器")
    minority = bundle.features["minority_signals"]
    if minority:
        st.dataframe(pd.DataFrame(minority), use_container_width=True, hide_index=True)
        evidence_comments(
            bundle,
            [item["comment_id"] for item in minority],
            "minority-comment",
            "少数意見の根拠",
        )
    else:
        st.info("条件を満たす少数意見は検出されませんでした。無理な生成はしていません。")

    st.subheader("F13 セマンティック・フレーズ")
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
            use_container_width=True,
        )
        st.plotly_chart(
            px.scatter(
                phrase_frame,
                x="score",
                y="phrase",
                size="count",
                color="dominant_emotion",
            ),
            use_container_width=True,
        )
        selected_phrase = st.selectbox("フレーズを選択", [item["phrase"] for item in phrases])
        selected = next(item for item in phrases if item["phrase"] == selected_phrase)
        evidence_comments(bundle, selected["comment_ids"], "phrase-comments")


def render_emotion(bundle: AnalysisBundle) -> None:
    st.header("感情・拡散")
    timeline = bundle.features["emotion_timeline"]
    st.subheader("F06 感情地震計")
    st.info(
        "実時間による推移です。"
        if timeline["axis_mode"] == "time"
        else "実時間ではなく投稿順による推移です。"
    )
    if timeline["bins"]:
        emotions = st.multiselect(
            "表示する感情",
            list(EMOTION_LABELS),
            default=["anger", "anxiety", "empathy", "hope", "moral_outrage"],
            format_func=lambda name: EMOTION_LABELS[name],
        )
        weighted = st.toggle("共感数で加重", value=False)
        smoothed = st.toggle("指数移動平均で平滑化", value=True)
        key = "weighted_mean" if weighted else "mean"
        long_rows = [
            {
                "位置": item["index"],
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
            timeline_frame, x="位置", y="強度", color="感情", markers=True
        )
        for point in timeline["change_points"]:
            fig.add_vline(x=point["position"], line_dash="dash", line_color="#d62728")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"自動ビン幅: {timeline['window_size']}コメント / "
            f"急変点: {len(timeline['change_points'])}件"
        )
        if timeline["change_points"]:
            cp_index = st.selectbox(
                "急変点",
                range(len(timeline["change_points"])),
                format_func=lambda index: (
                    f"位置 {timeline['change_points'][index]['position']} / "
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
                "急変付近の代表コメント",
            )
        else:
            st.info("設定感度で急変点は検出されませんでした。")
        st.caption(timeline["note"])

    st.subheader("F08 同調・言い換え拡散マップ")
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
        st.plotly_chart(graph, use_container_width=True)
        st.dataframe(edge_frame, use_container_width=True, hide_index=True)
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
    st.header("議論品質")
    st.subheader("F14 分断・健全性ダッシュボード")
    health = bundle.features["health"]
    health_frame = pd.DataFrame(
        [{"指標": key, "値": value} for key, value in health["metrics"].items()]
    )
    st.plotly_chart(
        px.bar(health_frame, x="値", y="指標", orientation="h", range_x=[0, 100]),
        use_container_width=True,
    )
    st.caption(f"信頼度 {health['confidence'] * 100:.0f}% — {health['warning']}")
    with st.expander("指標定義"):
        st.write(
            "多様性=クラスター分布、分断度=クラスター分離の推定、本文関連性=本文との意味類似、"
            "根拠提示率=理由・出典表現、建設性=提案表現、重複度=完全一致と言い換え、"
            "敬意=攻撃表現の少なさ、少数意見可視性=小規模意見群の比率、橋渡し率=中間密度です。"
        )

    st.subheader("F09 コメント品質フロンティア")
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
    x_axis = left.selectbox("X軸", dimensions, index=1)
    y_axis = right.selectbox("Y軸", dimensions, index=2)
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
            use_container_width=True,
        )
        evidence_comments(
            bundle, quality["comment_id"].tolist(), "quality-comments", "点を選択"
        )
    st.caption("共感数は点サイズにのみ使用し、品質スコアとは区別しています。")

    st.subheader("F10 レトリック・認知バイアスレンズ")
    rhetoric = bundle.features["rhetoric_lens"]
    threshold = st.slider("表示確率の下限", 0.60, 1.00, 0.60, 0.05)
    visible = [
        item for item in rhetoric["items"] if item["probability"] >= threshold
    ]
    if visible:
        st.dataframe(pd.DataFrame(visible), use_container_width=True, hide_index=True)
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
