from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class StanceType(str, Enum):
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    AMICUS = "amicus"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class ImportanceLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ExtractedPoint(BaseModel):
    #Key point extracted by LLM
    summary: str = Field(..., min_length=10, max_length=500)
    importance: ImportanceLevel
    importance_score: float = Field(..., ge=0.0, le=1.0)
    stance: StanceType
    supporting_quote: str
    legal_concepts: List[str] = Field(default_factory=list)
    page_reference: Optional[int] = None

class LLMAnalysisOutput(BaseModel):
    #Complete LLM analysis
    extracted_points: List[ExtractedPoint] = Field(..., min_length=1, max_length=15)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

class FinalKeyPoint(BaseModel):
    #Final key point with all metadata
    summary: str
    importance: ImportanceLevel
    importance_score: float
    stance: StanceType
    supporting_quote: str
    legal_concepts: List[str]
    page_number: Optional[int]
    section_type: Optional[str]
    retrieval_score: float = 0.0
    reranker_score: float = 0.0
    final_rank: int

class AnalysisResult(BaseModel):
    #Complete analysis result
    case_name: Optional[str] = None
    document_type: Optional[str] = None
    key_points: List[FinalKeyPoint]
    total_pages: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
