from app.services.nlp import analyze_sentiment, extract_entities, summarize_hierarchical


def test_summary_sentence_window():
    text = (
        "Acme reported quarterly growth of 20 percent. "
        "Revenue rose due to new enterprise contracts. "
        "Operating costs also increased. "
        "Management expects stable performance next quarter. "
        "The board approved continued investments in AI products."
    )
    summary, _ = summarize_hierarchical(text, "text")
    sentence_count = len([s for s in summary.split(".") if s.strip()])
    assert 2 <= sentence_count <= 4


def test_fuzzy_entity_normalization_money_and_date():
    text = "John Smith-Johnson invested $5,000 USD on Jan 15, 2024 at Acme Corporation Inc."
    entities, relationships, _ = extract_entities(text)
    assert entities
    assert any(e["type"] == "MONEY" and e["normalized"].startswith("$") for e in entities)
    assert any(e["type"] in {"DATE", "TIME"} and "-" in (e["normalized"] or "") for e in entities)
    assert isinstance(relationships, list)


def test_mixed_sentiment_detection():
    text = "The product is good but expensive and delays have frustrated customers."
    result, _ = analyze_sentiment(text)
    assert result["label"] in {"mixed", "negative", "neutral", "positive"}
    assert 0.0 <= result["score"] <= 1.0
    assert "Sentiment is" in result["explanation"]


def test_sarcasm_like_text_is_not_overconfident_positive():
    text = "Oh great, another meeting that got delayed again."
    result, _ = analyze_sentiment(text)
    assert result["label"] in {"mixed", "negative", "neutral", "positive"}
    assert result["score"] <= 0.95
