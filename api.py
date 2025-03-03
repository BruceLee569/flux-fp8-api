import os
from typing import Literal, Optional, TYPE_CHECKING

import requests
from tqdm import tqdm
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from platform import system

if TYPE_CHECKING:
    from flux_pipeline import FluxPipeline

if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1


class AppState:
    model: "FluxPipeline"


class FastAPIApp(FastAPI):
    state: AppState


class LoraArgs(BaseModel):
    scale: Optional[float] = 1.0
    path: Optional[str] = None
    name: Optional[str] = None
    action: Optional[Literal["load", "unload"]] = "load"


class LoraLoadResponse(BaseModel):
    status: Literal["success", "error"]
    message: Optional[str] = None


class GenerateArgs(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=720)
    height: Optional[int] = Field(default=1024)
    num_steps: Optional[int] = Field(default=24)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(
        default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND
    )
    strength: Optional[float] = 1.0
    init_image: Optional[str] = None


app = FastAPIApp()


@app.post("/generate")
def generate(args: GenerateArgs):
    """
    Generates an image from the Flux flow transformer.

    Args:
        args (GenerateArgs): Arguments for image generation:

            - `prompt`: The prompt used for image generation.

            - `width`: The width of the image.

            - `height`: The height of the image.

            - `num_steps`: The number of steps for the image generation.

            - `guidance`: The guidance for image generation, represents the
                influence of the prompt on the image generation.

            - `seed`: The seed for the image generation.

            - `strength`: strength for image generation, 0.0 - 1.0.
                Represents the percent of diffusion steps to run,
                setting the init_image as the noised latent at the
                given number of steps.

            - `init_image`: Base64 encoded image or path to image to use as the init image.

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    """
    result = app.state.model.generate(**args.model_dump())
    return StreamingResponse(result, media_type="image/jpeg")


@app.post("/lora", response_model=LoraLoadResponse)
def lora_action(args: LoraArgs):
    """
    Loads or unloads a LoRA checkpoint into / from the Flux flow transformer.

    Args:
        args (LoraArgs): Arguments for the LoRA action:

            - `scale`: The scaling factor for the LoRA weights.
            - `path`: The path to the LoRA checkpoint.
            - `name`: The name of the LoRA checkpoint.
            - `action`: The action to perform, either "load" or "unload".

    Returns:
        LoraLoadResponse: The status of the LoRA action.
    """
    try:
        if args.action == "load":
            app.state.model.load_lora(args.path, args.scale, args.name)
        elif args.action == "unload":
            app.state.model.unload_lora(args.name if args.name else args.path)
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Invalid action, expected 'load' or 'unload', got {args.action}",
                },
                status_code=400,
            )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )
    return JSONResponse(status_code=200, content={"status": "success"})

@app.on_event("startup")
def startup_event():

    lora_path = "/root/autodl-tmp/flux-fp8-api/loras/F.1_dev-fp8-lyf-12.safetensors"
    if not os.path.exists(lora_path):
        url = "https://github.com/BruceLee569/PublicSource/releases/download/v1.0.0/F.1_dev-fp8-lyf-12.safetensors"
        os.makedirs(os.path.dirname(lora_path), exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()
        # 获取文件大小（如果服务器提供）
        total_size = int(response.headers.get('content-length', 0))
        # 下载并显示进度条
        with open(lora_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"文件已下载并保存到: {save_path}")

    app.state.model.load_lora(lora_path, scale=1.2)

    payload = {
        "width": 512,
        "height": 512,
        "prompt": "a beautiful asian woman",
    }
    result = app.state.model.generate(**payload)
    payload = {
        "width": 768,
        "height": 768,
        "prompt": "a beautiful asian woman",
    }
    result = app.state.model.generate(**payload)
    payload = {
        "width": 1024,
        "height": 1024,
        "prompt": "a beautiful asian woman",
    }
    result = app.state.model.generate(**payload)

    print(f'首次加载预热：{result}')
