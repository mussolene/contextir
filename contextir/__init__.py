from contextir.clients import ModelResponse, OllamaClient, OpenAICompatibleClient
from contextir.gateway import ContextKind, ContractCheck, ContextBundle, ContextIR, SIRKernel, load_contextir, load_kernel
from contextir.pipeline import (
    ChunkLimitExceeded,
    ContextPipeline,
    ContextWindowExceeded,
    PipelinePolicy,
    PipelineResult,
    PreparedContext,
    ResponseVerification,
)

__version__ = "1.4.0"

__all__ = [
    "ContextIR",
    "ContextKind",
    "ChunkLimitExceeded",
    "ContextPipeline",
    "ContextWindowExceeded",
    "ModelResponse",
    "OllamaClient",
    "OpenAICompatibleClient",
    "PipelinePolicy",
    "PipelineResult",
    "PreparedContext",
    "ResponseVerification",
    "ContractCheck",
    "ContextBundle",
    "SIRKernel",
    "__version__",
    "load_contextir",
    "load_kernel",
]
