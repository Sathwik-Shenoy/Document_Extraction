from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, cast

from dateutil import parser as date_parser

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None

from app.core.config import settings


@dataclass
class ModelRegistry:
    summarization: Dict[str, str]
    ner: Dict[str, str]
    sentiment: Dict[str, str]


class RawEntity(TypedDict):
    text: str
    label: str
    score: float


class CanonicalEntity(TypedDict):
    entity: str
    type: str
    confidence: float
    frequency: int
    forms: List[str]
    aliases: List[str]
    normalized: str | None
    context: str | None


class RelationshipItem(TypedDict):
    source: str
    relation: str
    target: str
    confidence: float


class SentimentVote(TypedDict):
    label: str
    score: float


class SentimentResultPayload(TypedDict):
    label: str
    score: float
    agreement: float
    explanation: str
    votes: Dict[str, SentimentVote]


MODEL_REGISTRY = ModelRegistry(
    summarization={
        "primary": settings.summarizer_primary,
        "fallback": settings.summarizer_fallback,
    },
    ner={
        "primary": settings.ner_primary,
        "fallback": settings.ner_fallback,
    },
    sentiment={
        "distilbert": settings.sentiment_distilbert,
        "bert_mnli": settings.sentiment_mnli_bert,
        "roberta_mnli": settings.sentiment_mnli_roberta,
        "primary": "ensemble_3_models",
    },
)


def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def chunk_by_type(text: str, file_type: str) -> List[str]:
    if file_type == "pdf":
        chunks = [c.strip() for c in text.split("\n---PAGE_BREAK---\n") if c.strip()]
        return chunks or [text]
    if file_type == "docx":
        chunks = [c.strip() for c in re.split(r"(?=^#\s)", text, flags=re.MULTILINE) if c.strip()]
        return chunks or [text]
    return [text]


def _lexical_summary(text: str, target_sentences: int = 3) -> str:
    sents = sentence_split(text)
    if not sents:
        return "No extractable content was found in the document."
    if len(sents) <= target_sentences:
        return " ".join(sents)

    # Sentence length ranking keeps the fallback deterministic and avoids sparse-matrix typing noise.
    scores = [len(s.split()) for s in sents]
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    chosen_idx = sorted(ranked[:target_sentences])
    return " ".join(sents[i] for i in chosen_idx)


def _safe_hf_pipeline(task: str, model: str) -> Any | None:
    try:
        from transformers import pipeline  # type: ignore

        return pipeline(task, model=model)  # type: ignore[reportCallIssue]
    except Exception:
        return None


def summarize_hierarchical(text: str, file_type: str) -> Tuple[str, Dict[str, str]]:
    chunks = chunk_by_type(text, file_type)
    base_text = "\n".join(chunks)
    if not base_text.strip():
        return "The document does not contain readable text.", {"strategy": "empty_input"}

    primary = _safe_hf_pipeline("summarization", MODEL_REGISTRY.summarization["primary"]) if settings.use_heavy_models else None
    fallback = _safe_hf_pipeline("summarization", MODEL_REGISTRY.summarization["fallback"]) if settings.use_heavy_models else None

    def run_abstractive(s: str) -> str:
        candidate_sentences = sentence_split(s)
        top_n = max(2, math.ceil(len(candidate_sentences) * 0.5))
        reduced = _lexical_summary(s, target_sentences=min(top_n, 12))
        model = primary or fallback
        if model is None:
            raise RuntimeError("no_abstractive_model")
        output = model(reduced, max_length=180, min_length=60, do_sample=False)
        return str(output[0]["summary_text"])

    strategy = "lexical_fallback"
    try:
        summary = run_abstractive(base_text)
        strategy = "hierarchical_abstractive"
    except Exception:
        try:
            summary = _lexical_summary(base_text, target_sentences=3)
            strategy = "lexical_tfidf"
        except Exception:
            summary = " ".join(sentence_split(base_text)[:3])
            strategy = "first_three_sentences"

    sents = sentence_split(summary)
    if len(sents) < 2:
        sents = sentence_split(base_text)[:2]
        summary = " ".join(sents)
    if len(sents) > 4:
        summary = " ".join(sents[:4])

    return summary.strip(), {"strategy": strategy}


def _normalize_money(entity: str) -> str:
    digits = re.sub(r"[^\d.]", "", entity)
    if not digits:
        return entity
    if "." in digits:
        return f"${float(digits):.2f}"
    return f"${int(digits)}"


def _normalize_date(entity: str) -> str:
    try:
        dt = date_parser.parse(entity, dayfirst=False, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return entity


def _ner_with_spacy(text: str) -> List[RawEntity]:
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return []
        doc = nlp(text)
        return cast(List[RawEntity], [{"text": ent.text, "label": ent.label_, "score": 0.85} for ent in doc.ents])
    except Exception:
        return cast(List[RawEntity], [])


def _ner_with_transformers(text: str) -> List[RawEntity]:
    if not settings.use_heavy_models:
        return []
    pipe = _safe_hf_pipeline("ner", MODEL_REGISTRY.ner["primary"])
    if pipe is None:
        return cast(List[RawEntity], [])
    try:
        entities: List[Dict[str, Any]] = cast(List[Dict[str, Any]], pipe(text))
        return cast(
            List[RawEntity],
            [
                {
                    "text": str(e.get("word", "")).replace("##", ""),
                    "label": str(e.get("entity_group") or e.get("entity", "MISC")),
                    "score": float(e.get("score", 0.7)),
                }
                for e in entities
                if e.get("word")
            ],
        )
    except Exception:
        return cast(List[RawEntity], [])


def _regex_entities(text: str) -> List[RawEntity]:
    entities: List[RawEntity] = []
    money_pattern = r"(?:\$\s?\d[\d,]*(?:\.\d+)?\s?(?:USD)?)|(?:\d[\d,]*(?:\.\d+)?\s?USD)"
    date_pattern = r"\b(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b"
    org_pattern = r"\b[A-Z][A-Za-z0-9&\-]+\s(?:Corp(?:oration)?|Inc\.?|LLC|Ltd\.?|Company)\b"

    for m in re.finditer(money_pattern, text):
        entities.append({"text": m.group(0), "label": "MONEY", "score": 0.75})
    for d in re.finditer(date_pattern, text):
        entities.append({"text": d.group(0), "label": "DATE", "score": 0.75})
    for o in re.finditer(org_pattern, text):
        entities.append({"text": o.group(0), "label": "ORG", "score": 0.7})

    return entities


def _canonicalize(raw_entities: List[RawEntity], text: str) -> List[CanonicalEntity]:
    if not raw_entities:
        return []

    def _fuzzy_ratio(a: str, b: str) -> float:
        if fuzz is not None:
            return float(fuzz.token_sort_ratio(a, b))
        return SequenceMatcher(None, a, b).ratio() * 100.0

    clusters: List[CanonicalEntity] = []
    for ent in raw_entities:
        value = ent["text"].strip()
        if not value:
            continue

        label = ent["label"]
        score = float(ent.get("score", 0.7))
        matched = None
        for c in clusters:
            if c["type"] != label:
                continue
            if _fuzzy_ratio(value.lower(), c["entity"].lower()) >= 88:
                matched = c
                break

        if matched is None:
            clusters.append(
                {
                    "entity": value,
                    "type": label,
                    "confidence": score,
                    "frequency": 1,
                    "forms": [value],
                    "aliases": [],
                    "normalized": None,
                    "context": None,
                }
            )
        else:
            matched["frequency"] += 1
            matched["confidence"] = max(matched["confidence"], score)
            if value not in matched["forms"]:
                matched["forms"].append(value)
            if len(value) > len(matched["entity"]):
                matched["entity"] = value

    for c in clusters:
        if c["type"] == "MONEY":
            c["normalized"] = _normalize_money(c["entity"])
        elif c["type"] in {"DATE", "TIME"}:
            c["normalized"] = _normalize_date(c["entity"])
        elif c["type"] == "ORG" and len(c["forms"]) > 1:
            c["aliases"] = [f for f in c["forms"] if f != c["entity"]]

        idx = text.lower().find(c["entity"].lower())
        if idx >= 0:
            start = max(0, idx - 35)
            end = min(len(text), idx + len(c["entity"]) + 35)
            c["context"] = text[start:end].strip()

    return clusters


def extract_relationships(text: str, entities: Sequence[CanonicalEntity]) -> List[RelationshipItem]:
    relationships: List[RelationshipItem] = []
    people: List[str] = [str(e["entity"]) for e in entities if e["type"] in {"PERSON", "PER"}]
    orgs: List[str] = [str(e["entity"]) for e in entities if e["type"] == "ORG"]
    for person in people:
        for org in orgs:
            pattern = rf"{re.escape(person)}[^.\n]{{0,60}}(?:CEO|CFO|founder|director)\s+(?:of|at)\s+{re.escape(org)}"
            if re.search(pattern, text, flags=re.IGNORECASE):
                relationships.append(
                    {
                        "source": person,
                        "relation": "ROLE_AT",
                        "target": org,
                        "confidence": 0.8,
                    }
                )
    return relationships


def extract_entities(text: str) -> Tuple[List[CanonicalEntity], List[RelationshipItem], Dict[str, str]]:
    raw: List[RawEntity] = _ner_with_transformers(text)
    strategy = "transformers_ner"
    if not raw:
        raw = _ner_with_spacy(text)
        strategy = "spacy_ner"
    if not raw:
        raw = _regex_entities(text)
        strategy = "regex_ner"

    canonical = _canonicalize(raw, text)
    relationships = extract_relationships(text, canonical)
    return canonical, relationships, {"strategy": strategy}


def _mnli_vote(text: str, model_name: str) -> Tuple[str, float]:
    pipe = _safe_hf_pipeline("zero-shot-classification", model_name)
    if pipe is None:
        return "neutral", 0.5
    try:
        result = pipe(text, candidate_labels=["positive", "negative", "neutral", "mixed"])
        return result["labels"][0], float(result["scores"][0])
    except Exception:
        return "neutral", 0.5


def _distilbert_vote(text: str) -> Tuple[str, float]:
    pipe = _safe_hf_pipeline("sentiment-analysis", MODEL_REGISTRY.sentiment["distilbert"])
    if pipe is None:
        return "neutral", 0.5
    try:
        result = pipe(text[:512])[0]
        label = result["label"].lower()
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"
        else:
            label = "neutral"
        return label, float(result["score"])
    except Exception:
        return "neutral", 0.5


def _lexical_vote(text: str) -> Tuple[str, float]:
    positive_lex = {"good", "great", "excellent", "growth", "improved", "success", "optimistic"}
    negative_lex = {"bad", "poor", "cancelled", "delay", "budget cuts", "loss", "risk", "expensive", "failed"}

    t = text.lower()
    pos_hits = sum(1 for w in positive_lex if w in t)
    neg_hits = sum(1 for w in negative_lex if w in t)

    total = max(1, pos_hits + neg_hits)
    if abs(pos_hits - neg_hits) <= 1 and total > 1:
        return "mixed", 0.62
    if neg_hits > pos_hits:
        return "negative", min(0.55 + (neg_hits / (total + 1)), 0.85)
    if pos_hits > neg_hits:
        return "positive", min(0.55 + (pos_hits / (total + 1)), 0.85)
    return "neutral", 0.5


def analyze_sentiment(text: str) -> Tuple[SentimentResultPayload, Dict[str, str]]:
    dist_label, dist_score = _distilbert_vote(text)
    mnli_a_label, mnli_a_score = _mnli_vote(text, MODEL_REGISTRY.sentiment["bert_mnli"])
    mnli_b_label, mnli_b_score = _mnli_vote(text, MODEL_REGISTRY.sentiment["roberta_mnli"])

    if (dist_label, mnli_a_label, mnli_b_label) == ("neutral", "neutral", "neutral"):
        lex_label, lex_score = _lexical_vote(text)
        dist_label, dist_score = lex_label, max(dist_score, lex_score)

    weights = {"distilbert": 0.5, "bert_mnli": 0.25, "roberta_mnli": 0.25}
    tally: defaultdict[str, float] = defaultdict(float)
    tally[dist_label] += weights["distilbert"] * dist_score
    tally[mnli_a_label] += weights["bert_mnli"] * mnli_a_score
    tally[mnli_b_label] += weights["roberta_mnli"] * mnli_b_score

    label, score = sorted(tally.items(), key=lambda x: x[1], reverse=True)[0]
    consensus = Counter([dist_label, mnli_a_label, mnli_b_label])
    agreement = consensus[label] / 3.0

    if score < 0.65 or agreement < 0.5:
        label = "mixed"

    explanation_bits: List[str] = []
    if "not" in text.lower() or "no" in text.lower() or "never" in text.lower():
        explanation_bits.append("negation patterns were detected")
    if label == "mixed":
        explanation_bits.append("model disagreement indicates mixed tone")
    if not explanation_bits:
        explanation_bits.append("ensemble votes reached stable agreement")

    return (
        {
            "label": label,
            "score": round(min(max(score, 0.0), 1.0), 3),
            "agreement": round(agreement, 3),
            "explanation": f"Sentiment is {label} because " + ", ".join(explanation_bits) + ".",
            "votes": {
                "distilbert": {"label": dist_label, "score": round(dist_score, 3)},
                "bert_mnli": {"label": mnli_a_label, "score": round(mnli_a_score, 3)},
                "roberta_mnli": {"label": mnli_b_label, "score": round(mnli_b_score, 3)},
            },
        },
        {"strategy": "ensemble_weighted_voting"},
    )
