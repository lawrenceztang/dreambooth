---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
widget:

        - text: 'Lawrence’s Face'
        
base_model: stabilityai/stable-diffusion-xl-base-1.0
instance_prompt: Lawrence’s Face
license: openrail++
---

# SDXL LoRA DreamBooth - lora-dreambooth-model

<Gallery />

## Model description

### These are lora-dreambooth-model LoRA adaption weights for stabilityai/stable-diffusion-xl-base-1.0.

## Download model

### Use it with UIs such as AUTOMATIC1111, Comfy UI, SD.Next, Invoke

- **LoRA**: download **[`lora-dreambooth-model.safetensors` here 💾](/lora-dreambooth-model/blob/main/lora-dreambooth-model.safetensors)**.
    - Place it on your `models/Lora` folder.
    - On AUTOMATIC1111, load the LoRA by adding `<lora:lora-dreambooth-model:1>` to your prompt. On ComfyUI just [load it as a regular LoRA](https://comfyanonymous.github.io/ComfyUI_examples/lora/).


## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('lora-dreambooth-model', weight_name='pytorch_lora_weights.safetensors')

image = pipeline('Lawrence’s Face').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## Trigger words

You should use Lawrence’s Face to trigger the image generation.

## Details
All [Files & versions](/lora-dreambooth-model/tree/main).

The weights were trained using [🧨 diffusers Advanced Dreambooth Training Script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py).

LoRA for the text encoder was enabled. False.

Pivotal tuning was enabled: False.

Special VAE used for training: None.

