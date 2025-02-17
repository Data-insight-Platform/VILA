import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from typing import Union
from contextlib import asynccontextmanager
import logging

from llava.utils import disable_torch_init
from llava.utils.logging import logger
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from serve.config import parse_args
from serve.models import HealthCheckResponse, ErrorResponse, ChatCompletionRequest
from serve.service import handle_request


settings = parse_args()
model = None
tokenizer = None
image_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, model, tokenizer, image_processor
    disable_torch_init()
    model_name = get_model_name_from_path(settings.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        settings.model_path,
        model_name,
        None,
        num_video_frames=settings.num_video_frames,
        model_max_length=settings.model_max_length,
        max_sequence_length=settings.max_sequence_length,
    )
    logger.info(f"Model loaded: {model_name}. Context length: {context_len}")
    yield


app = FastAPI(
    title="VILA/NVILA Service",
    description="API for VILA and NVILA model inference",
    version="1.0.0",
    lifespan=lifespan,
)

# region OPENAI_COMPATIBILITY


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests"""
    try:
        if request.model != service.model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {service.model_name}, "
                f"but the request model is {request.model}"
            )
        return await handle_request(request, model, tokenizer, image_processor, settings.conv_mode)
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(e)).dict())


# endregion


# region SAGEMAKER_COMPATIBILITY


@app.get("/ping", response_model=HealthCheckResponse)
async def ping():
    """SageMaker health check endpoint"""
    try:
        if model is None:
            return JSONResponse(
                status_code=500,
                content=HealthCheckResponse(status="Model not loaded").dict(),
            )
        return HealthCheckResponse(status="Healthy")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=HealthCheckResponse(status=f"Unhealthy: {str(e)}").dict(),
        )


@app.post("/invocations")
async def invocations(request: Request):
    """SageMaker invocation endpoint that maintains OpenAI compatibility"""
    try:
        body = await request.json()
        chat_request = ChatCompletionRequest(**body)
        return await chat_completions(chat_request)
    except Exception as e:
        logger.error(f"Invocation failed: {str(e)}")
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(e)).dict())


# endregion

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port,
                workers=settings.num_workers)
