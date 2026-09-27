from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 1. Target search domain (Memory / Knowledge / Dual-track)
class TargetDomain(str, Enum):
    MEMORY_ONLY = "memory"  # Short/long-term conversational memory, user persona
    KNOWLEDGE_ONLY = (
        "knowledge"  # Documents, technical documentation, wiki, knowledge base
    )
    BOTH = "both"  # Parallel retrieval across memory + knowledge, followed by fusion


# 2. Pipeline precision level (RAG 1.0 to 2.0)
class PipelineLevel(str, Enum):
    V1_DENSE = "v1_dense"  # Basic dense vector similarity
    V1_5_HYBRID = "v1_5_hybrid"  # Dense + Sparse hybrid retrieval
    V2_GRAPH_AWARE = (
        "v2_graph_aware"  # Hybrid + Knowledge Graph / Association traversal
    )


# 3. Output context format
class Granularity(str, Enum):
    RAW_CHUNK = "raw_chunk"
    PARENT_CHUNK = "parent_chunk"
    WINDOW_EXPANDED = "window_expanded"


# --- Domain-Specific Query Options ---


class MemoryQueryConfig(BaseModel):
    """Memory retrieval configuration options."""

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    enable_time_decay: bool = True  # Whether to apply recency weighting
    recency_weight: float = 0.3  # Time decay weight factor (0.0 to 1.0)
    importance_threshold: float = 0.5  # Minimum memory importance score cutoff
    reference_timestamp: Optional[datetime] = (
        None  # Reference timestamp (defaults to current time)
    )


class KnowledgeQueryConfig(BaseModel):
    """Knowledge base retrieval configuration options."""

    collection_names: List[str] = Field(default_factory=lambda: ["default_kb"])
    sparse_vector: Optional[Dict[str, float]] = (
        None  # SPLADE/BM25 token weight map
    )
    hybrid_alpha: float = (
        0.7  # Dense weight (1.0: Dense-only, 0.0: Sparse-only)
    )
    # RAG 2.0 Knowledge Graph traversal parameters
    seed_entities: Optional[List[str]] = None
    max_hops: int = 1


class DualTrackFusionConfig(BaseModel):
    """Fusion configuration when target_domain is set to BOTH."""

    memory_weight: float = 0.4  # Score weight for memory results
    knowledge_weight: float = 0.6  # Score weight for knowledge base results
    interleave_results: bool = (
        False  # Interleave results ([mem1, kb1, mem2...]) instead of score-based merge
    )


# --- Final Order Specification ---


class RAGQueryOrder(BaseModel):
    # Pipeline routing and target domain selection
    target_domain: TargetDomain = TargetDomain.BOTH
    pipeline_level: PipelineLevel = PipelineLevel.V1_5_HYBRID

    # Common dense embedding vector and retrieval limits.
    # None means "embed it yourself" — Fuli owns embedding now, so this is
    # only ever set when something local has already computed a vector.
    dense_vector: Optional[List[float]] = None
    top_k: int = 5
    min_score_threshold: float = 0.6

    # Domain-specific configurations (inject conditionally)
    memory_config: Optional[MemoryQueryConfig] = None
    knowledge_config: Optional[KnowledgeQueryConfig] = None
    fusion_config: DualTrackFusionConfig = Field(
        default_factory=DualTrackFusionConfig
    )

    # Post-processing and context shaping
    granularity: Granularity = Granularity.RAW_CHUNK
    context_window_size: int = 1
    enable_rerank: bool = False


# --- Response Specification (what Fuli sends back) ---


class RetrievedHit(BaseModel):
    """One retrieved past conversation turn from Fuli's memory store."""

    id: int
    user_input: str
    model_response: str
    score: float


class EmotionDelta(BaseModel):
    v: float
    a: float
    d: float
    r: float


class EmotionDynamics(BaseModel):
    delta: EmotionDelta
    lability: float


class EmotionCumulative(BaseModel):
    stress: float
    stress_ratio: float
    reward: float
    reward_ratio: float
    total: float


class EmotionFront(BaseModel):
    expression: str
    intensity: float
    similarity: float


class EmotionInstant(BaseModel):
    stress: float
    stress_ratio: float
    reward: float
    reward_ratio: float
    ratio_total: float
    deviation: float


class EmotionAnalysis(BaseModel):
    cumulative: EmotionCumulative
    dynamics: EmotionDynamics
    front: EmotionFront
    instant: EmotionInstant


class EmotionCurrentState(BaseModel):
    v: float  # valence
    a: float  # arousal
    d: float  # dominance
    r: float  # resistance (physics_weights.yaml's FormulaWeights.resistance)


class EmotionResult(BaseModel):
    """deltaEGO's full emotion read-out: a VAD+resistance state plus a human-readable label."""

    current_state: EmotionCurrentState
    emotion_term: str  # e.g. "penitent" — the mood tag to surface in prompts
    similarity: float
    analysis: EmotionAnalysis


class CharacterContextResponse(BaseModel):
    """
    The full body Fuli returns from POST /character/context, in response
    to a rag_policy + emotion_policy_raw request (see FuliHandler).
    """

    hits: List[RetrievedHit] = Field(default_factory=list)
    emotion: EmotionResult


# --- Save Specification (what Fuli sends back from the save/close-out call) ---


class SavedMemoryContent(BaseModel):
    user_input: str
    model_response: str


class SavedMemoryEmotion(BaseModel):
    current: EmotionCurrentState
    emotion_terms: List[str]
    similarity: float


class SavedMemoryPersona(BaseModel):
    persona_name: str
    persona_content: str


class SavedMemoryUserId(BaseModel):
    """A 128-bit hashed user id, split into two 64-bit halves."""

    high: int
    low: int


class SavedMemoryUser(BaseModel):
    user_name: str
    user_content: str
    user_id: SavedMemoryUserId


class SavedMemory(BaseModel):
    content: SavedMemoryContent
    emotion: SavedMemoryEmotion
    persona: SavedMemoryPersona
    user: SavedMemoryUser


class SavedMemoryMetadata(BaseModel):
    faiss_id: int
    session_id: str
    importance: float
    timestamp: int


class EmotionAnalysisWrapper(BaseModel):
    """deltaEGO's raw read-out at save time — same shape as CharacterContextResponse.emotion."""

    deltaEGO_analysis: EmotionResult


class ContextSaveResponse(BaseModel):
    """
    The full body Fuli returns from POST /character/context_memory
    (schemas::FuliContextSaveRequest in). `query` (an echo of the original
    context request) is accepted loosely since nothing consumes its exact
    shape yet.
    """

    memory: SavedMemory
    metadata: SavedMemoryMetadata
    emotion_analysis: Optional[EmotionAnalysisWrapper] = None
    query: Optional[Dict[str, Any]] = None

    def to_client_payload(self) -> Dict[str, Any]:
        """
        Sanitized view safe to forward to an edge Client: always drops
        `query` (a verbose echo of our original Fuli request — config,
        rag_policy, etc.) and `memory.user.user_id` (the hashed id) —
        neither should ever reach the frontend.
        """
        data = self.model_dump(exclude={"query"})
        data["memory"]["user"].pop("user_id", None)
        return data