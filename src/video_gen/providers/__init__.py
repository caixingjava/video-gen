"""External service clients used by the workflow agents."""

from .openai_client import OpenAIWorkflowClient
from .xunfei import XunfeiTTSClient
from .dashscope_music import DashscopeMusicClient
from .freesound import FreesoundClient

__all__ = [
    "OpenAIWorkflowClient",
    "XunfeiTTSClient",
    "DashscopeMusicClient",
    "FreesoundClient",
]
