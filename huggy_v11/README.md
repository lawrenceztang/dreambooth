---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
widget:

        - text: 'a <s0><s1> dressed as Yoda'
          output:
            url:
                "image_0.png"
        
        - text: 'a <s0><s1> dressed as Yoda'
          output:
            url:
                "image_1.png"
        
        - text: 'a <s0><s1> dressed as Yoda'
          output:
            url:
                "image_2.png"
        
        - text: 'a <s0><s1> dressed as Yoda'
          output:
            url:
                "image_3.png"
        
base_model: stabilityai/stable-diffusion-xl-base-1.0
instance_prompt: a photo of <s0><s1>
license: openrail++
---

# SDXL LoRA DreamBooth - lawrenceztang/huggy_v11

<Gallery />

## Model description

### These are lawrenceztang/huggy_v11 LoRA adaption weights for stabilityai/stable-diffusion-xl-base-1.0.

## Download model

### Use it with UIs such as AUTOMATIC1111, Comfy UI, SD.Next, Invoke

- **LoRA**: download **[`huggy_v11.safetensors` here 💾](/lawrenceztang/huggy_v11/blob/main/huggy_v11.safetensors)**.
    - Place it on your `models/Lora` folder.
    - On AUTOMATIC1111, load the LoRA by adding `<lora:huggy_v11:1>` to your prompt. On ComfyUI just [load it as a regular LoRA](https://comfyanonymous.github.io/ComfyUI_examples/lora/).
- *Embeddings*: download **[`huggy_v11_emb.safetensors` here 💾](/lawrenceztang/huggy_v11/blob/main/huggy_v11_emb.safetensors)**.
    - Place it on it on your `embeddings` folder
    - Use it by adding `huggy_v11_emb` to your prompt. For example, `a photo of huggy_v11_emb`
    (you need both the LoRA and the embeddings as they were trained together for this LoRA)
    

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
        
pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('lawrenceztang/huggy_v11', weight_name='pytorch_lora_weights.safetensors')
embedding_path = hf_hub_download(repo_id='lawrenceztang/huggy_v11', filename='huggy_v11_emb.safetensors' repo_type="model")
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
pipeline.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
        
image = pipeline('a <s0><s1> dressed as Yoda').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## Trigger words

To trigger image generation of trained concept(or concepts) replace each concept identifier in you prompt with the new inserted tokens:

to trigger concept `TOK` → use `<s0><s1>` in your prompt 



## Details
All [Files & versions](/lawrenceztang/huggy_v11/tree/main).

The weights were trained using [🧨 diffusers Advanced Dreambooth Training Script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py).

LoRA for the text encoder was enabled. False.

Pivotal tuning was enabled: True.

Special VAE used for training: madebyollin/sdxl-vae-fp16-fix.

