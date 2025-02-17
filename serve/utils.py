import requests
import re
import base64
from PIL import Image as PILImage
from PIL.Image import Image
from io import BytesIO
from llava.constants import DEFAULT_IMAGE_TOKEN


def load_image(image_url: str) -> Image:
    IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


def normalize_image_tags(qs: str) -> str:
    if DEFAULT_IMAGE_TOKEN not in qs:
        print("No image was found in input messages. Continuing with text only prompt.")
    return qs
