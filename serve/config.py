import os
from pydantic import BaseModel, Field, root_validator
from typing import Optional, Dict
from argparse import ArgumentParser

from llava.mm_utils import get_model_name_from_path
from llava.utils.logging import logger


class Settings(BaseModel):
    host: str = Field(description="Server host address")
    port: int = Field(description="Server port")
    num_workers: int = Field(description="Number of worker processes")
    model_path: str = Field(
        description="Path to the model",
        default=os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3B"),
    )
    model_name: Optional[str] = Field(description="Model name (optional)")
    context_length: Optional[int] = Field(description="Context length (optional)")
    conv_mode: str = Field(description="Conversation mode")
    num_video_frames: int = Field(description="Number of video frames", default=int(os.getenv("NUM_VIDEO_FRAMES", 8)))
    model_max_length: int = Field(description="Model max length", default=int(os.getenv("MODEL_MAX_LENGTH", 4096)))
    max_sequence_length: Optional[int] = Field(
        description="Max sequence length",
        default=os.getenv("MAX_SEQUENCE_LENGTH", None),
    )

    @root_validator(pre=False)
    def set_model_name(cls, values: Dict) -> Dict:
        values["model_name"] = get_model_name_from_path(values["model_path"])
        return values

    def show(self):
        try:
            from tabulate import tabulate
        except ImportError:
            logger.error("Please install tabulate to use this feature")
            return

        headers = ["Parameter", "Value"]
        data = self.dict()
        rows = [(k, v) for k, v in data.items()]
        print(tabulate(rows, headers=headers, tablefmt="pretty"))


def parse_args() -> Settings:
    parser = ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8080")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("serve", nargs="?", help="SageMaker serve argument")
    args = parser.parse_args()
    args.__dict__.pop("serve")
    return Settings(**args.__dict__)
