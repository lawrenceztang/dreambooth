from diffusers import AutoPipelineForText2Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import openai
import os
import sys
from openai import OpenAI

# Set your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def get_prompt_from_chatgpt():
    query = "Provide a short prompt (20 words) for generating a funny image. Include <s0><s1> as a token that represents the user who is a human and make sure their face is visible in the image."
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1.2
    )
    try:
        return response.choices[0].message.content
    except:
        return 'a computer and <s0><s1>'  # Fallback prompt

def generate_images(n=1, iterations=25):
    # Load the model and LORA weights
    pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                         torch_dtype=torch.float16).to('cuda')
    pipeline.load_lora_weights('lawrenceztang/huggy_v11', weight_name='pytorch_lora_weights.safetensors')

    # Load the textual inversion embeddings
    embedding_path = hf_hub_download(repo_id='lawrenceztang/huggy_v11', filename='huggy_v11_emb.safetensors', repo_type="model")
    state_dict = load_file(embedding_path)
    pipeline.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder,
                                    tokenizer=pipeline.tokenizer)
    pipeline.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder_2,
                                    tokenizer=pipeline.tokenizer_2)

    # Generate and save images
    for i in range(n):
        prompt = get_prompt_from_chatgpt()
        print(f"Using prompt for image {i + 1}: {prompt}")
        image = pipeline(prompt, num_inference_steps=iterations).images[0]
        image.save(f'outputs/output_{i}.png')
        print(f'Image {i + 1} saved as outputs/output_{i}.png')

if __name__ == "__main__":
    # Set default values
    default_n = 1
    default_iterations = 25

    # Parse arguments
    if len(sys.argv) > 1:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else default_n
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else default_iterations
    else:
        n = default_n
        iterations = default_iterations

    # Generate images
    generate_images(n, iterations)
