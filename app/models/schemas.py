from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EntityItem(BaseModel):
    entity: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    frequency: int = Field(ge=1)
    forms: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    normalized: Optional[str] = None
    context: Optional[str] = None


class RelationshipItem(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float = Field(ge=0.0, le=1.0)


class SentimentResult(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    votes: Dict[str, Dict[str, object]] = Field(default_factory=dict)


class ConfidenceScores(BaseModel):
    extraction: float = Field(ge=0.0, le=1.0)
    summary: float = Field(ge=0.0, le=1.0)
    entities: float = Field(ge=0.0, le=1.0)
    sentiment: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    document_id: str
    file_type: str
    extracted_text: str
    summary: str
    entities: List[EntityItem]
    relationships: List[RelationshipItem] = Field(default_factory=list)
    sentiment: SentimentResult
    confidence_scores: ConfidenceScores
    model_versions: Dict[str, str]
    metadata: Dict[str, object]
