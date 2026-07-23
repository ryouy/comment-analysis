from src.preprocessing.text import clean_text, extract_phrases, split_sentences


def test_clean_text_normalizes_and_removes_unsafe_noise() -> None:
    assert clean_text("ＡＢＣ <b>本文</b> https://example.com\u0001") == "ABC 本文"


def test_split_sentences_keeps_paragraph_position() -> None:
    result = split_sentences("最初です。次です。\n三つ目です。")
    assert [item[2] for item in result] == ["最初です。", "次です。", "三つ目です。"]
    assert result[-1][0] == 1


def test_phrase_extraction_excludes_stopwords_and_has_compounds() -> None:
    result = dict(extract_phrases(["政府の説明責任 政府の説明責任 こと"], limit=20))
    assert "こと" not in result
    assert any("・" in phrase for phrase in result)

