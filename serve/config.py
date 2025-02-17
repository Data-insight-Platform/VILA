import os
from pydantic import BaseModel, Field
from typing import Optional
from argparse import ArgumentParser


class Settings(BaseModel):
    host: str = Field(description="Server host address")
    port: int = Field(description="Server port")
    workers: int = Field(description="Number of worker processes")
    model_path: str = Field(description="Path to the model")
    model_name: Optional[str] = Field(description="Model name (optional)")
    context_length: Optional[int] = Field(description="Context length (optional)")
    conv_mode: str = Field(description="Conversation mode")
    num_video_frames: int = Field(
        description="Number of video frames", default=os.getenv("NUM_VIDEO_FRAMES", 8)
    )
    model_max_length: int = Field(
        description="Model max length", default=os.getenv("MODEL_MAX_LENGTH", 4096)
    )
    max_sequence_length: Optional[int] = Field(
        description="Max sequence length", default=os.getenv("MAX_SEQUENCE_LENGTH", None)
    )


def parse_args() -> Settings:
    parser = ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8080")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-path", default="Efficient-Large-Model/VILA1.5-3B")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("serve", nargs="?", help="SageMaker serve argument")
    args = parser.parse_args()
    args.__dict__.pop("serve")
    return Settings(**args.__dict__)
