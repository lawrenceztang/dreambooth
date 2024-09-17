from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('lora-dreambooth-model', weight_name='pytorch_lora_weights.safetensors')

image = pipeline('Lawrence’s Face').images[0]
image.save("output.png")
