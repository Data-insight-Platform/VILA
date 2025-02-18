import time
from uuid import uuid4
from typing import Callable, List, Dict
from fastapi.responses import JSONResponse

from llava.media import Video
from llava.utils.logging import logger
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.constants import DEFAULT_IMAGE_TOKEN


from serve.models import ChatCompletionRequest, ChatMessage
from serve.utils import load_image, normalize_image_tags


def parse_messages(conv: Conversation, messages: List, user_role: str, assistant_role: str):
    images = []
    for message in messages:
        if message.role == "user":
            text_content = []
            image_tokens = []
            media_content = []

            # Process each content piece in the message
            for content in message.content:
                if isinstance(content, str):
                    text_content.append(content)
                elif hasattr(content, "type"):
                    if content.type == "text":
                        text_content.append(content.text)
                    elif content.type == "image_url":
                        url = content.image_url.url
                        if any(url.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
                            image = load_image(url)
                            images.append(image)
                            image_tokens.append(DEFAULT_IMAGE_TOKEN)
                        else:
                            raise ValueError(f"Unsupported image format: {url}")
                    elif content.type == "video_url":
                        url = content.video_url.url
                        if any(url.endswith(ext) for ext in (".mp4", ".mkv", ".webm")):
                            media_content.append(Video(url))
                        else:
                            raise ValueError(f"Unsupported video format: {url}")
                    else:
                        raise ValueError(f"Unsupported media type: {content.type}")
                else:
                    raise ValueError(f"Invalid content format: {content}")

            final_content = []
            if text_content:
                final_content.extend(text_content)
            if image_tokens:
                final_content.extend(normalize_image_tags(image_tokens))
            if media_content:
                final_content.extend(media_content)

            conv.append_message(user_role, final_content)

        elif message.role == "assistant":
            content = message.content
            if isinstance(content, list):
                content = [p.text if hasattr(p, "text") else str(p) for p in content]
                content = [c for c in content if c]  # Filter out empty strings
            if content:  # Only append if there's content
                conv.append_message(assistant_role, content)

    return images, media_content


def image_inference(images, model, tokenizer, image_processor):
    if len(images) == 0:
        images_input = None
    else:
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        images_input = [images_tensor]

    input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        if request.stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)
            thread = Thread(
                target=model.generate,
                kwargs=dict(
                    input_ids=input_ids,
                    images=images_input,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    use_cache=use_cache,
                    stopping_criteria=[stopping_criteria],
                ),
            )
            thread.start()

            def chunk_generator():
                prepend_space = False
                should_stop = False
                chunk_id = 0
                for new_text in streamer:
                    if new_text == " ":
                        prepend_space = True
                        continue
                    if new_text.endswith(stop_str):
                        new_text = new_text[: -len(stop_str)].strip()
                        prepend_space = False
                        should_stop = True
                    elif prepend_space:
                        new_text = " " + new_text
                        prepend_space = False
                    if len(new_text):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": time.time(),
                            "choices": [{"delta": {"content": new_text}}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(chunk_generator())

        else:
            output_ids = model.generate(
                input_ids,
                images=images_input,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_tokens,
                use_cache=use_cache,
                stopping_criteria=[stopping_criteria],
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            print("\nAssistant: ", outputs)

            resp_content = [TextContent(type="text", text=outputs)]
            return {
                "id": uuid4().hex,
                "object": "chat.completion",
                "created": time.time(),
                "choices": [{"message": ChatMessage(role="assistant", content=resp_content)}],
            }


def video_inference(model, prompt) -> Dict:
    logger.info(f"Recieved prompt: {prompt}")
    response = model.generate_content(prompt)
    return {
        "id": uuid4().hex,
        "object": "chat.completion",
        "created": time.time(),
        "choices": [{"message": ChatMessage(role="assistant", content=response)}],
    }


async def handle_request(
    request: ChatCompletionRequest, model: Callable, tokenizer: Callable, image_processor: Callable, conv_mode: str
) -> JSONResponse:
    messages = request.messages
    conv = conv_templates[conv_mode].copy()
    assistant_role = conv.roles[1]
    images, videos = parse_messages(conv, messages, conv.roles[0], assistant_role)

    if conv.sep_style == SeparatorStyle.LLAMA_3:
        conv.append_message(assistant_role, "")

    if len(images) > 0:
        prompt_text = conv.get_prompt()
        print("Prompt input: ", prompt_text)
        return image_inference(images, model, tokenizer, image_processor)
    if len(videos) > 0:
        _, prompt = conv.messages[0]
        return video_inference(model, prompt)
