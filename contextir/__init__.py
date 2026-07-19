from contextir.clients import ModelResponse, OllamaClient, OpenAICompatibleClient
from contextir.gateway import ContractCheck, ContextBundle, ContextIR, SIRKernel, load_contextir, load_kernel
from contextir.pipeline import ContextPipeline, PipelinePolicy, PipelineResult, PreparedContext, ResponseVerification

__version__ = "1.0.1"

__all__ = [
    "ContextIR",
    "ContextPipeline",
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
