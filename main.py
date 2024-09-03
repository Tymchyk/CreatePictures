from fastapi import FastAPI, Form, Query,HTTPException
from fastapi.responses import JSONResponse
import torch
import gc
import numpy as np
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLPipeline,StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import base64
import logging
import torch.nn.functional as F
import cv2
from insightface.app import FaceAnalysis
from skimage import io
from huggingface_hub import hf_hub_download
from controlnet_aux import NormalBaeDetector
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
import base64
import logging
import torch.nn.functional as F
import cv2
from insightface.app import FaceAnalysis
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import base64
from skimage import io
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import random
import numpy as np
import asyncio
import base64
from io import BytesIO
import logging
from asyncio import Lock, Queue
from typing import Dict
import uuid
from insightface.utils import face_align
from diffusers import DiffusionPipeline
#Upscaler
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'face-swap'))
from roop.core import start
from roop.face_analyser import get_one_face
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import roop.globals
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import uuid
import subprocess
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging
import cv2

# Папки для збереження вхідних та вихідних файлів
PRETRAINED_WEIGHTS_DIR = "./LivePortrait/LivePortrait/pretrained_weights"
INPUT_DIR = "./input"
OUTPUT_DIR = "./animations"  # стандартний вихідний каталог для інференс скрипта
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AnimationRequest(BaseModel):
    source_image: str
    driving_video: str

class AnimationResponse(BaseModel):
    result_video_base64: str

# Допоміжні функції
def save_base64_to_file(data: str, output_path: str):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(data))

def read_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def resize_image(input_path: str, output_path: str, width: int = 256, height: int = 256):
    img = cv2.imread(input_path)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)

class FaceSwapRequest(BaseModel):
    source_image: str
    target_image: str
    doFaceEnhancer: bool = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Модель 1: Завантаження моделей для генерації зображень
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")

# Налаштування шумового планувальника
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# Завантаження VAE
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
# Завантаження основної моделі
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
# Завантаження IP-адаптера
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
# Модель 2: RMBG v1.4 для видалення фону
rmbg_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
rmbg_model.to(device)

# Модель 3: ControlNet для спеціальних ефектів
controlnet_model_id = "lllyasviel/control_v11p_sd15_normalbae"
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to(device)

#base_model_url = "https://huggingface.co/broyang/hentaidigitalart_v20/blob/main/realcartoon3d_v15.safetensors"

#controlnet_pipe = StableDiffusionControlNetPipeline.from_single_file(
    #base_model_url,
    #controlnet=controlnet,
    #scheduler=noise_scheduler,
    #safety_checker=None,
    #torch_dtype=torch.float16,
#).to(device)

preprocessor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(device)
MODELS = {
    "UltraSharp": "Kim2091/UltraSharp",
    "DAT X4": "OzzyGT/DAT_X4",
    "DAT X3": "OzzyGT/DAT_X3",
    "DAT X2": "OzzyGT/DAT_X2",
    "RealPLKSR X4": "OzzyGT/4xNomosWebPhoto_RealPLKSR",
    "DAT-2 RealWebPhoto X4": "Phips/4xRealWebPhoto_v4_dat2",
}

class UpscaleImageRequest(BaseModel):
    image_base64: str
    model_selection: str

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Зменшення розміру зображення до max_size x max_size з дотриманням пропорцій."""
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def convert_to_jpeg_and_compress(image: Image.Image, quality: int = 100) -> str:
    """Конвертація зображення в JPEG і стиснення до вказаної якості."""
    buffer = BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Функції попередньої і пост-обробки
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

pipe_v4 = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16
).to(device)

def make_divisible_by_8(value):
    return (value // 8) * 8

style_list = [
    {
        "name": "3d-model",
        "prompt": "ultra-realistic professional 3D model of {prompt}. rendered in octane, hyper-detailed textures, volumetric lighting, realistic materials, photorealistic, intricate design, high resolution, dynamic shadows",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting, low detail, flat, simplistic",
    },
    {
        "name": "anime",
        "prompt": "high-quality anime artwork of {prompt}. vibrant colors, dynamic poses, studio anime level, highly detailed characters, expressive features, sharp lines, rich background elements",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast, muted colors, simplistic",
    },
    {
        "name": "cinematic",
        "prompt": "cinematic still of {prompt}. emotional scene, harmonious composition, vignette effect, highly detailed, high budget production, dramatic lighting, bokeh effect, widescreen, moody atmosphere, epic scale, stunning visuals, film grain texture",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "digital-art",
        "prompt": "high-quality digital artwork of {prompt}. exceptionally detailed, professional standard, sharp focus, vibrant colors, intricate designs, imaginative themes, crisp lines, polished finish, rich textures",
        "negative_prompt": "blurry, low resolution, deformed, ugly, flat colors, simplistic, low detail, unprofessional",
    },
    {
        "name": "fantasy-art",
        "prompt": "imaginative fantasy artwork of {prompt}. magical elements, ethereal atmosphere, highly detailed, vibrant colors, mystical creatures, intricate designs, epic landscapes, fantastical themes, dynamic composition",
        "negative_prompt": "blurry, deformed, low detail, ugly, flat colors, mundane, uninspired, simplistic",
    },
    {
        "name": "isometric",
        "prompt": "highly detailed isometric illustration of {prompt}. clean lines, vibrant colors, professional quality, intricate designs, 3D perspective, polished finish, clear geometry, rich details",
        "negative_prompt": "blurry, deformed, low detail, ugly, flat perspective, simplistic, unprofessional",
    },
    {
        "name": "line-art",
        "prompt": "intricate line art of {prompt}. clean lines, highly detailed, professional quality, sharp focus, complex designs, high contrast, dynamic composition, polished finish, rich details",
        "negative_prompt": "blurry, deformed, low detail, ugly, messy lines, low contrast, flat, simplistic",
    },
    {
        "name": "low-poly",
        "prompt": "highly detailed low poly illustration of {prompt}. geometric shapes, clean lines, vibrant colors, intricate details, 3D effect, professional quality, complex designs, polished finish, rich textures",
        "negative_prompt": "blurry, deformed, low detail, ugly, simplistic shapes, flat colors, unprofessional, low resolution",
    },
    {
        "name": "neon-punk",
        "prompt": "highly detailed neon punk illustration of {prompt}. vibrant neon colors, futuristic elements, professional quality, intricate designs, sharp lines, cyberpunk themes, glowing effects, dynamic composition",
        "negative_prompt": "blurry, deformed, low detail, ugly, muted colors, outdated, simplistic, unprofessional",
    },
    {
        "name": "logo",
        "prompt": "professional logo of {prompt}. clean design, highly detailed, sharp focus, vibrant colors, crisp lines, modern aesthetics, polished finish, clear symbolism, intricate details",
        "negative_prompt": "blurry, deformed, low detail, ugly, messy, outdated, unclear, simplistic",
    },
    {
        "name": "realistic",
        "prompt": "hyper-realistic depiction of {prompt}. highly detailed, professional quality, sharp focus, lifelike textures, accurate proportions, natural colors, intricate details, polished finish, rich textures",
        "negative_prompt": "blurry, deformed, low detail, ugly, unrealistic, cartoonish, flat, simplistic",
    },
    {
        "name": "photo",
        "prompt": "cinematic photo of {prompt}. 35mm photograph quality, film-like appearance, bokeh effect, professional lighting, 4k resolution, highly detailed, natural colors, sharp focus, intricate textures, realistic composition",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "pixel-art",
        "prompt": "high-quality pixel art of {prompt}. clean lines, highly detailed, vibrant colors, professional quality, crisp pixels, intricate designs, retro style, polished finish, rich details",
        "negative_prompt": "blurry, deformed, low detail, ugly, messy pixels, low contrast, flat, simplistic",
    },
    {
        "name": "No style",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

request_queue = asyncio.Queue()
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests())

@app.get("/generate-from-prompt")
async def generate_image(
    prompt: str,
    style: str = Query("(No style)"),
    num_inference_steps: int = Query(20, gt=0, le=50),
    guidance_scale: float = Query(7.5, gt=0, le=15),
    width: int = Query(768, gt=0, le=1024),
    height: int = Query(768, gt=0, le=1024),
    num_images: int = Query(1, gt=0, le=10)
):
    # Create a unique request ID
    request_id = random.randint(0, 1000000)
    request_data = {
        "id": request_id,
        "prompt": prompt,
        "style": style,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "num_images": num_images
    }
    await request_queue.put(request_data)
    logger.info(f"Request {request_id} added to the queue")
    
    # Wait until the request is processed
    while True:
        result = await asyncio.sleep(1)
        if request_data.get("result"):
            break
    return JSONResponse(content=request_data["result"])

async def process_requests():
    while True:
        request_data = await request_queue.get()
        try:
            width = make_divisible_by_8(request_data["width"])
            height = make_divisible_by_8(request_data["height"])

            style_data = next((item for item in style_list if item["name"].lower() == request_data["style"].lower()), None)
            if not style_data:
                logger.error(f"Invalid style: {request_data['style']}")
                raise HTTPException(status_code=400, detail="Invalid style")

            prompt = style_data["prompt"].format(prompt=request_data["prompt"])
            negative_prompt = style_data["negative_prompt"]

            seed = random.randint(0, 1000000)
            generator = torch.manual_seed(seed)
            num_images = request_data["num_images"]
            options = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": request_data["num_inference_steps"],
                "guidance_scale": request_data["guidance_scale"],
                "generator": generator,
                "output_type": "pil",
                "negative_prompt": negative_prompt,
                "num_images_per_prompt":num_images
            }

            logger.info(f"Generating image with options: {options}")

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipe_v4(**options)
            )
            torch.cuda.empty_cache()
            gc.collect()
            images = result.images
            encoded_images = []

            for image in images:
               buffered = BytesIO()
               image.save(buffered, format="WEBP", quality=95, method=6)
               img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
               encoded_images.append(img_str)

            logger.info(f"Image generation successful for request {request_data['id']}")
            request_data["result"] = {"images": encoded_images}
        except Exception as e:
            torch.cuda.empty_cache()
            logger.error(f"Error during image generation for request {request_data['id']}: {str(e)}")
            request_data["result"] = {"error": str(e)}
        finally:
            request_queue.task_done()

@app.get("/health")
async def health_check():
    """Перевірка працездатності сервера"""
    return JSONResponse(content={"status": "ok"})

@app.post("/generate/")
async def generate_images(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_samples: int = Form(1),
    width: int = Form(768),
    height: int = Form(768),
    num_inference_steps: int = Form(25),
    seed: int = Form(2023),
    image_base64: str = Form(None)
):
    logger.info(f"Received request with prompt: {prompt}")
    logger.info(f"Negative prompt: {negative_prompt}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Width: {width}, Height: {height}")
    logger.info(f"Number of inference steps: {num_inference_steps}")
    logger.info(f"Seed: {seed}")
    torch.cuda.empty_cache()
    if image_base64:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")
            return JSONResponse(content={"error": "Error processing image data"}, status_code=400)
    else:
        return JSONResponse(content={"error": "No image data provided"}, status_code=400)

    try:
        app_face = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_face.prepare(ctx_id=0, det_size=(640, 640))
        faces = app_face.get(image)
        if not faces:
            logger.warning("No faces detected")
            return JSONResponse(content={"error": "No faces detected"}, status_code=400)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
    except Exception as e:
        logger.error(f"Error in face analysis: {str(e)}")
        return JSONResponse(content={"error": "Error in face analysis"}, status_code=400)

    try:
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, no duplicate, avoid repetition"
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt = negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_samples,
            width=width,
            height=height,
            num_inference_steps=30,
        )
        logger.info("Image generation successful")
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return JSONResponse(content={"error": "Error during image generation"}, status_code=500)

    try:
        image_list = []
        for img in images:
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            image_list.append(base64.b64encode(img_byte_arr.getvalue()).decode('utf-8'))
        logger.info("Images successfully encoded and sent")
        return JSONResponse(content={"images": image_list})
    except Exception as e:
        logger.error(f"Error encoding images: {str(e)}")
        return JSONResponse(content={"error": "Error encoding images"}, status_code=500)

@app.post("/remove_background/")
async def remove_background(
    image_base64: str = Form(...)
):
    logger.info("Received request for background removal")
    if not image_base64:
        return JSONResponse(content={"error": "No image data provided"}, status_code=400)

    try:
        # Декодування зображення
        image_data = base64.b64decode(image_base64)
        orig_im = Image.open(BytesIO(image_data)).convert("RGB")
        orig_im_size = orig_im.size
        logger.info(f"Original image size: {orig_im_size}")
        orig_im = np.array(orig_im)

        model_input_size = [1024, 1024]

        # Попередня обробка
        image_tensor = preprocess_image(orig_im, model_input_size).to(device)
        logger.info(f"Image tensor size after preprocessing: {image_tensor.shape}")

        # Інференс
        with torch.no_grad():
            result = rmbg_model(image_tensor)

        # Постобробка
        result_image = postprocess_image(result[0][0], orig_im_size)
        logger.info(f"Result image size after postprocessing: {result_image.shape}")
        # Створення зображення без фону
        pil_im = Image.fromarray(result_image).convert("L")
        logger.info(f"Mask size: {pil_im.size}")
        if pil_im.size != orig_im_size:
             logger.info(f"Adjusting mask size from {pil_im.size} to {orig_im_size}")
             pil_im = pil_im.resize(orig_im_size, Image.Resampling.LANCZOS)
             logger.info(f"Adjusted mask size: {pil_im.size}")
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        orig_image = Image.open(BytesIO(image_data)).convert("RGBA")

        no_bg_image.paste(orig_image, mask=pil_im)
        logger.info(f"Final image size: {no_bg_image.size}")

        # Конвертація зображення без фону в base64
        img_byte_arr = BytesIO()
        no_bg_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        logger.info("Background removal successful")
        return JSONResponse(content={"image": encoded_image})

    except Exception as e:
        logger.error(f"Error during background removal: {str(e)}")
        return JSONResponse(content={"error": f"Error during background removal: {str(e)}"}, status_code=500)

@app.post("/generate_controlnet/")
async def generate_controlnet_images(
    image_base64: str = Form(...),
    prompt: str = Form(""),
    additional_prompt: str = Form(""),
    negative_prompt: str = Form(""),
    num_images: int = Form(1),
    image_resolution: int = Form(512),
    preprocess_resolution: int = Form(768),
    num_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    seed: int = Form(0)
):
    logger.info("Received request for ControlNet image generation")
    if not image_base64:
        return JSONResponse(content={"error": "No image data provided"}, status_code=400)

    try:
        # Декодування зображення
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Попередня обробка
        control_image = preprocessor(
            input_image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
        )

        # Генерація
        generator = torch.cuda.manual_seed(seed)
        if len(prompt) >= 1:
              custom_prompt = prompt + "extremely detailed in unity 8k wallpaper, ultra high quality, 4k resolution, photorealistic, masterpiece, best quality, highly detailed, facing forward, facing the viewer, as similar as possible to the provided image"
        else:
              custom_prompt = "hyperrealistic photography,extremely detailed,(intricate details),unity 8k wallpaper,ultra detailed,tungsten white balance"
        result = controlnet_pipe(
            prompt=custom_prompt,
            negative_prompt=negative_prompt + "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=num_steps,
            generator=generator,
            image=control_image,
        ).images[0]

        # Конвертація результату в base64
        img_byte_arr = BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        logger.info("ControlNet image generation successful")
        return JSONResponse(content={"image": encoded_image})

    except Exception as e:
        logger.error(f"Error during ControlNet image generation: {str(e)}")
        return JSONResponse(content={"error": f"Error during ControlNet image generation: {str(e)}"}, status_code=500)


@app.post("/upscale-image/")
async def upscale_image(request: UpscaleImageRequest):
    try:
        logger.info("Received upscale request.")

        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            logger.info("Image decoded successfully.")
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

        try:
            # Завантаження моделі на основі вибору користувача
            model_selection = request.model_selection
            if model_selection not in MODELS:
                raise HTTPException(status_code=400, detail="Invalid model selection.")

            upscaler = UpscaleWithModel.from_pretrained(MODELS[model_selection]).to("cuda")

            # Обробка зображення
            original = load_image(image)
            output = upscaler(original, tiling=True, tile_width=1024, tile_height=1024)

            buffered = BytesIO()
            output_image = Image.fromarray(np.array(output))
            resized_image = resize_image(output_image, max_size=1024)

            # Конвертація у JPEG та стиснення
            compressed_image_base64 = convert_to_jpeg_and_compress(resized_image)

            logger.info("Image processed successfully.")
            return {"image": compressed_image_base64}

        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail="Failed to process image.")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/swap-face/")
async def swap_face(request: FaceSwapRequest):
    session_id = str(uuid.uuid4())
    session_dir = f"temp/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    try:
        source_image_data = base64.b64decode(request.source_image)
        target_image_data = base64.b64decode(request.target_image)

        source_image = Image.open(BytesIO(source_image_data)).convert("RGB")
        target_image = Image.open(BytesIO(target_image_data)).convert("RGB")

        source_path = os.path.join(session_dir, "input.jpg")
        target_path = os.path.join(session_dir, "target.jpg")

        source_image.save(source_path)
        target_image.save(target_path)

        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            raise HTTPException(status_code=400, detail="No face in source path detected.")

        target_face = get_one_face(cv2.imread(target_path))
        if target_face is None:
            raise HTTPException(status_code=400, detail="No face in target path detected.")

        output_path = os.path.join(session_dir, "output.jpg")
        normalized_output_path = normalize_output_path(source_path, target_path, output_path)

        frame_processors = ["face_swapper", "face_enhancer"] if request.doFaceEnhancer else ["face_swapper"]

        for frame_processor in get_frame_processors_modules(frame_processors):
            if not frame_processor.pre_check():
                raise HTTPException(status_code=500, detail=f"Pre-check failed for {frame_processor}")
        roop.globals.source_path = source_path
        roop.globals.target_path = target_path
        roop.globals.output_path = normalized_output_path
        roop.globals.frame_processors = frame_processors
        roop.globals.headless = True
        roop.globals.keep_fps = True
        roop.globals.keep_audio = True
        roop.globals.keep_frames = False
        roop.globals.many_faces = False
        roop.globals.video_encoder = "libx264"
        roop.globals.video_quality = 18
        roop.globals.execution_providers = ['CUDAExecutionProvider']
        roop.globals.reference_face_position = 0
        roop.globals.similar_face_distance = 0.6
        roop.globals.max_memory = 24
        roop.globals.execution_threads = 50

        start()

        with open(normalized_output_path, "rb") as output_file:
            encoded_output_image = base64.b64encode(output_file.read()).decode("utf-8")
        return {"output_image": encoded_output_image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        #Видалення тимчасових файлів
        if os.path.exists(session_dir):
            for file_name in os.listdir(session_dir):
                file_path = os.path.join(session_dir, file_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            os.rmdir(session_dir)

@app.post("/animate/", response_model=AnimationResponse)
async def animate_image_and_video(request: AnimationRequest):
    try:
        # Унікальний ідентифікатор для цієї сесії
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(INPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Шляхи до тимчасових файлів
        source_image_path = os.path.join(session_dir, "source_image.jpg")
        driving_video_path = os.path.join(session_dir, "driving_video.mp4")

        # Збереження завантажених файлів
        save_base64_to_file(request.source_image, source_image_path)
        save_base64_to_file(request.driving_video, driving_video_path)

        # Виконання команди для запуску інференсу
        command = [
            "python3", "./LivePortrait/LivePortrait/inference.py",  # Оновлено шлях до скрипта
            "-s", source_image_path,
            "-d", driving_video_path
        ]
        subprocess.run(command, check=True)

        # Пошук вихідних файлів
        result_files = [
            "source_image--driving_video.mp4",
            "source_image--driving_video_concat.mp4"
        ]

        for result_file in result_files:
            result_path = os.path.join(OUTPUT_DIR, result_file)
            if os.path.exists(result_path):
                result_video_base64 = read_file_to_base64(result_path)
                return AnimationResponse(result_video_base64=result_video_base64)

        raise HTTPException(status_code=500, detail="No output file found after inference.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Inference process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference process failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Очищення тимчасових файлів
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
