from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


# region Request Models
class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url", "video_url"]
    image_url: Optional[ImageURL] = None
    video_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions"""

    model: Literal[
        "NVILA-8B",
        "nvila-8b",
        "NVILA-8B-Lite",
        "nvila-8b-lite",
        "NVILA-15B",
        "nvila-15b",
        "NVILA-15B-Lite",
        "nvila-15b-lite",
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
    ] = Field(..., description="The model to use for completion")
    messages: List[ChatMessage] = Field(..., description="The messages to generate completions for")
    max_tokens: Optional[int] = Field(512, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling probability threshold")
    temperature: Optional[float] = Field(0.2, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    use_cache: Optional[bool] = Field(True, description="Whether to use the cache")
    num_beams: Optional[int] = Field(1, description="Number of beams for beam search")


# endregion

# region Response Models


class ChatCompletionChunk(BaseModel):
    """Model for streaming response chunks"""

    id: str = Field(..., description="Unique identifier for the chunk")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Unix timestamp of when the chunk was created")
    model: str = Field(..., description="The model used for completion")
    choices: List[dict] = Field(..., description="List of completion choices")


class ChatCompletionResponse(BaseModel):
    """Model for chat completion responses"""

    id: str = Field(..., description="Unique identifier for the response")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp of when the response was created")
    model: str = Field(..., description="The model used for completion")
    choices: List[dict] = Field(..., description="List of completion choices")


class HealthCheckResponse(BaseModel):
    """Model for health check response"""

    status: str = Field(..., description="Health status of the service")


class ErrorResponse(BaseModel):
    """Model for error responses"""

    error: str = Field(..., description="Error message")


# endregion
