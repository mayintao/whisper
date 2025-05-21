import uuid

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    safety_checker=None,#禁用全检查器
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

# 不设置随机种子，或者每次都用不同的随机种子
generator = torch.Generator("cpu").manual_seed(torch.randint(0, 1_000_000, (1,)).item())

result = pipe(
    prompt="Korean beautiful, college girl, good face, young, yoga",
    negative_prompt="blurry, low quality, text, watermark",
    height=512,
    width=512,
    num_inference_steps=30,#控制图像生成的细节和质量，10~100之间，40是比较常用的平衡点
    guidance_scale=8.5,#控制模型生成时对提示词的依赖强度，5~15，8.5是一个较为常见且平衡的默认值
    num_images_per_prompt=2,  # 👈 几张图
    generator=generator
)

for idx, img in enumerate(result.images):
    unique_name = f"output_{uuid.uuid4().hex}.png"
    img.save(unique_name)
