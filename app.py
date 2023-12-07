import gradio as gr
from PIL import Image
import requests
import subprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from huggingface_hub import snapshot_download, HfApi
import torch
import uuid
import os
import shutil
import json
import random
from slugify import slugify
import argparse 
import importlib
import sys
from pathlib import Path
MAX_IMAGES = 50

training_script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py"
subprocess.run(['wget', training_script_url])

device = "cuda" if torch.cuda.is_available() else "cpu"

FACES_DATASET_PATH = snapshot_download(repo_id="multimodalart/faces-prior-preservation", repo_type="dataset")

#Delete .gitattributes to process things properly
Path(FACES_DATASET_PATH, '.gitattributes').unlink(missing_ok=True)


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
)
#Run first captioning as apparently makes the other ones faster
pil_image = Image.new('RGB', (512, 512), 'black')
blip_inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**blip_inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def load_captioning(uploaded_images, option):
    updates = []
    if len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(
            f"Error: for now, only {MAX_IMAGES} or less images are allowed for training"
        )
    # Update for the captioning_area
    for _ in range(3):
        updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        text_value = option if visible else None
        updates.append(gr.update(value=text_value, visible=visible))
    return updates

def check_removed_and_restart(images):
    visible = bool(images)
    return [gr.update(visible=visible) for _ in range(3)]

def make_options_visible(option):
    if (option == "object") or (option == "face"):
        sentence = "A photo of TOK"
    elif option == "style":
        sentence = "in the style of TOK"
    elif option == "custom":
        sentence = "TOK"
    return (
        gr.update(value=sentence, visible=True),
        gr.update(visible=True),
    )
def change_defaults(option, images):
    num_images = len(images)
    max_train_steps = num_images * 150
    max_train_steps = 500 if max_train_steps < 500 else max_train_steps
    random_files = []
    with_prior_preservation = False
    class_prompt = ""
    if(num_images > 24):
        repeats = 1
    elif(num_images > 10):
        repeats = 2
    else:
        repeats = 3
    if(max_train_steps > 2400):
        max_train_steps = 2400
        
    if(option == "face"):
        rank = 64
        max_train_steps = num_images*100
        lr_scheduler = "constant"
        #Takes 150 random faces for the prior preservation loss
        directory = FACES_DATASET_PATH
        file_count = 150
        files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]       
        random_files = random.sample(files, min(len(files), file_count))
        with_prior_preservation = True
        class_prompt = "a photo of a person"
    elif(option == "style"):
        rank = 16
        lr_scheduler = "polynomial"
    elif(option == "object"):
        rank = 8
        repeats = 1
        lr_scheduler = "constant"
    else:
        rank = 32
        lr_scheduler = "constant"
        
    return max_train_steps, repeats, lr_scheduler, rank, with_prior_preservation, class_prompt, random_files

def create_dataset(*inputs):
    print("Creating dataset")
    images = inputs[0]
    destination_folder = str(uuid.uuid4())
    print(destination_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    jsonl_file_path = os.path.join(destination_folder, 'metadata.jsonl')
    with open(jsonl_file_path, 'a') as jsonl_file:
        for index, image in enumerate(images):
            new_image_path = shutil.copy(image, destination_folder)
            
            original_caption = inputs[index + 1]
            file_name = os.path.basename(new_image_path)

            data = {"file_name": file_name, "prompt": original_caption}

            jsonl_file.write(json.dumps(data) + "\n")
    
    return destination_folder

def start_training(
    lora_name,
    training_option,
    concept_sentence,
    optimizer,
    use_snr_gamma,
    snr_gamma,
    mixed_precision,
    learning_rate,
    train_batch_size,
    max_train_steps,
    lora_rank,
    repeats,
    with_prior_preservation,
    class_prompt,
    class_images,
    num_class_images,
    train_text_encoder_ti,
    train_text_encoder_ti_frac,
    num_new_tokens_per_abstraction,
    train_text_encoder,
    train_text_encoder_frac,
    text_encoder_learning_rate,
    seed,
    resolution,
    num_train_epochs,
    checkpointing_steps,
    prior_loss_weight,
    gradient_accumulation_steps,
    gradient_checkpointing,
    enable_xformers_memory_efficient_attention,
    adam_beta1,
    adam_beta2,
    prodigy_beta3,
    prodigy_decouple,
    adam_weight_decay,
    adam_weight_decay_text_encoder,
    adam_epsilon,
    prodigy_use_bias_correction,
    prodigy_safeguard_warmup,
    max_grad_norm,
    scale_lr,
    lr_num_cycles,
    lr_scheduler,
    lr_power,
    lr_warmup_steps,
    dataloader_num_workers,
    local_rank,
    dataset_folder,
    token,
    progress = gr.Progress(track_tqdm=True)
): 
    print("Started training")
    slugged_lora_name = slugify(lora_name)
    spacerunner_folder = str(uuid.uuid4())
    commands = [
        "pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"instance_prompt={concept_sentence}",
        f"dataset_name=./{dataset_folder}",
        "caption_column=prompt",
        f"output_dir={slugged_lora_name}",
        f"mixed_precision={mixed_precision}",
        f"resolution={int(resolution)}",
        f"train_batch_size={int(train_batch_size)}",
        f"repeats={int(repeats)}",
        f"gradient_accumulation_steps={int(gradient_accumulation_steps)}",
        f"learning_rate={learning_rate}",
        f"text_encoder_lr={text_encoder_learning_rate}",
        f"adam_beta1={adam_beta1}",
        f"adam_beta2={adam_beta2}",
        f"optimizer={'adamW' if optimizer == '8bitadam' else optimizer}",
        f"train_text_encoder_ti_frac={train_text_encoder_ti_frac}",
        f"lr_scheduler={lr_scheduler}",
        f"lr_warmup_steps={int(lr_warmup_steps)}",
        f"rank={int(lora_rank)}",
        f"max_train_steps={int(max_train_steps)}",
        f"checkpointing_steps={int(checkpointing_steps)}",
        f"seed={int(seed)}",
        f"prior_loss_weight={prior_loss_weight}",
        f"num_new_tokens_per_abstraction={int(num_new_tokens_per_abstraction)}",
        f"num_train_epochs={int(num_train_epochs)}",
        f"prodigy_beta3={prodigy_beta3}",
        f"adam_weight_decay={adam_weight_decay}",
        f"adam_weight_decay_text_encoder={adam_weight_decay_text_encoder}",
        f"adam_epsilon={adam_epsilon}",
        f"prodigy_decouple={prodigy_decouple}",
        f"prodigy_use_bias_correction={prodigy_use_bias_correction}",
        f"prodigy_safeguard_warmup={prodigy_safeguard_warmup}",
        f"max_grad_norm={max_grad_norm}",
        f"lr_num_cycles={int(lr_num_cycles)}",
        f"lr_power={lr_power}",
        f"dataloader_num_workers={int(dataloader_num_workers)}",
        f"local_rank={int(local_rank)}",
        "cache_latents",
        "push_to_hub",
    ]
    # Adding optional flags
    if optimizer == "8bitadam":
        commands.append("use_8bit_adam")
    if gradient_checkpointing:
        commands.append("gradient_checkpointing")
    
    if train_text_encoder_ti:
        commands.append("train_text_encoder_ti")
    elif train_text_encoder:
        commands.append("train_text_encoder")
        commands.append(f"--train_text_encoder_frac={train_text_encoder_frac}")
    if enable_xformers_memory_efficient_attention: 
        commands.append("enable_xformers_memory_efficient_attention")
    if use_snr_gamma: 
        commands.append(f"snr_gamma={snr_gamma}")
    if scale_lr:
        commands.append("scale_lr")
    if with_prior_preservation:
        commands.append("with_prior_preservation")
        commands.append(f"class_prompt={class_prompt}")
        commands.append(f"num_class_images={int(num_class_images)}")
        if class_images:
            class_folder = str(uuid.uuid4())
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            for image in class_images:
                shutil.copy(image, class_folder)
            commands.append(f"class_data_dir={class_folder}")
            shutil.copytree(class_folder, f"{spacerunner_folder}/{class_folder}")
    # Joining the commands with ';' separator for spacerunner format
    spacerunner_args = ';'.join(commands)
    if not os.path.exists(spacerunner_folder):
        os.makedirs(spacerunner_folder)
    shutil.copy("train_dreambooth_lora_sdxl_advanced.py", f"{spacerunner_folder}/script.py")
    shutil.copytree(dataset_folder, f"{spacerunner_folder}/{dataset_folder}")
    requirements='''-peft
torch
git+https://github.com/huggingface/diffusers@c05d71be04345b18a5120542c363f6e4a3f99b05
transformers
accelerate
safetensors
prodigyopt
hf-transfer
git+https://github.com/huggingface/datasets.git'''
    file_path = f'{spacerunner_folder}/requirements.txt'
    with open(file_path, 'w') as file:
        file.write(requirements)
    # The subprocess call for autotrain spacerunner
    api = HfApi(token=token)
    username = api.whoami()["name"]
    subprocess_command = ["autotrain", "spacerunner", "--project-name", slugged_lora_name, "--script-path", spacerunner_folder, "--username", username, "--token", token, "--backend", "spaces-a10gl", "--env","HF_TOKEN=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;HF_HUB_ENABLE_HF_TRANSFER=1", "--args", spacerunner_args]
    print(subprocess_command)
    subprocess.run(subprocess_command)
    return f"<h2>Your training has started. Run over to <a href='https://huggingface.co/spaces/{username}/autotrain-{slugged_lora_name}?logs=container'>{username}/autotrain-{slugged_lora_name}</a> to check the status (click the logs tab)</h2>"

def start_training_og(
    lora_name,
    training_option,
    concept_sentence,
    optimizer,
    use_snr_gamma,
    snr_gamma,
    mixed_precision,
    learning_rate,
    train_batch_size,
    max_train_steps,
    lora_rank,
    repeats,
    with_prior_preservation,
    class_prompt,
    class_images,
    num_class_images,
    train_text_encoder_ti,
    train_text_encoder_ti_frac,
    num_new_tokens_per_abstraction,
    train_text_encoder,
    train_text_encoder_frac,
    text_encoder_learning_rate,
    seed,
    resolution,
    num_train_epochs,
    checkpointing_steps,
    prior_loss_weight,
    gradient_accumulation_steps,
    gradient_checkpointing,
    enable_xformers_memory_efficient_attention,
    adam_beta1,
    adam_beta2,
    prodigy_beta3,
    prodigy_decouple,
    adam_weight_decay,
    adam_weight_decay_text_encoder,
    adam_epsilon,
    prodigy_use_bias_correction,
    prodigy_safeguard_warmup,
    max_grad_norm,
    scale_lr,
    lr_num_cycles,
    lr_scheduler,
    lr_power,
    lr_warmup_steps,
    dataloader_num_workers,
    local_rank,
    dataset_folder,
    progress = gr.Progress(track_tqdm=True)
):
    slugged_lora_name = slugify(lora_name)
    print(train_text_encoder_ti_frac)
    commands = ["--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
            "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
            f"--instance_prompt={concept_sentence}",
            f"--dataset_name=./{dataset_folder}",
            "--caption_column=prompt",
            f"--output_dir={slugged_lora_name}",
            f"--mixed_precision={mixed_precision}",
            f"--resolution={int(resolution)}",
            f"--train_batch_size={int(train_batch_size)}",
            f"--repeats={int(repeats)}",
            f"--gradient_accumulation_steps={int(gradient_accumulation_steps)}",
            f"--learning_rate={learning_rate}",
            f"--text_encoder_lr={text_encoder_learning_rate}",
            f"--adam_beta1={adam_beta1}",
            f"--adam_beta2={adam_beta2}",
            f"--optimizer={'adamW' if optimizer == '8bitadam' else optimizer}",
            f"--train_text_encoder_ti_frac={train_text_encoder_ti_frac}",
            f"--lr_scheduler={lr_scheduler}",
            f"--lr_warmup_steps={int(lr_warmup_steps)}",
            f"--rank={int(lora_rank)}",
            f"--max_train_steps={int(max_train_steps)}",
            f"--checkpointing_steps={int(checkpointing_steps)}",
            f"--seed={int(seed)}",
            f"--prior_loss_weight={prior_loss_weight}",
            f"--num_new_tokens_per_abstraction={int(num_new_tokens_per_abstraction)}",
            f"--num_train_epochs={int(num_train_epochs)}",
            f"--prodigy_beta3={prodigy_beta3}",
            f"--adam_weight_decay={adam_weight_decay}",
            f"--adam_weight_decay_text_encoder={adam_weight_decay_text_encoder}",
            f"--adam_epsilon={adam_epsilon}",
            f"--prodigy_decouple={prodigy_decouple}",
            f"--prodigy_use_bias_correction={prodigy_use_bias_correction}",
            f"--prodigy_safeguard_warmup={prodigy_safeguard_warmup}",
            f"--max_grad_norm={max_grad_norm}",
            f"--lr_num_cycles={int(lr_num_cycles)}",
            f"--lr_power={lr_power}",
            f"--dataloader_num_workers={int(dataloader_num_workers)}",
            f"--local_rank={int(local_rank)}",
            "--cache_latents"
            ]
    if optimizer == "8bitadam":
        commands.append("--use_8bit_adam")
    if gradient_checkpointing:
        commands.append("--gradient_checkpointing")
    
    if train_text_encoder_ti:
        commands.append("--train_text_encoder_ti")
    elif train_text_encoder:
        commands.append("--train_text_encoder")
        commands.append(f"--train_text_encoder_frac={train_text_encoder_frac}")
    if enable_xformers_memory_efficient_attention: 
        commands.append("--enable_xformers_memory_efficient_attention")
    if use_snr_gamma: 
        commands.append(f"--snr_gamma={snr_gamma}")
    if scale_lr:
        commands.append("--scale_lr")
    if with_prior_preservation:
        commands.append(f"--with_prior_preservation")
        commands.append(f"--class_prompt={class_prompt}")
        commands.append(f"--num_class_images={int(num_class_images)}")
        if(class_images):
            class_folder = str(uuid.uuid4())
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            for image in class_images:
                shutil.copy(image, class_folder)
            commands.append(f"--class_data_dir={class_folder}")

    print(commands)
    from train_dreambooth_lora_sdxl_advanced import main as train_main, parse_args as parse_train_args
    args = parse_train_args(commands)
    train_main(args)
    #print(commands)
    #subprocess.run(commands)
    return "ok!"

def run_captioning(*inputs):
    print(inputs)
    images = inputs[0]
    training_option = inputs[-1]
    print(training_option)
    final_captions = [""] * MAX_IMAGES
    for index, image in enumerate(images):
        original_caption = inputs[index + 1]
        pil_image = Image.open(image)  
        blip_inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**blip_inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if training_option == "style":
            final_caption = generated_text + " " + original_caption
        else:
            final_caption = original_caption + " " + generated_text
        final_captions[index] = final_caption
        yield final_captions

def check_token(token):
    try:
        api = HfApi(token=token)
    except Exception as e:
        gr.Warning("Invalid user token. Make sure to get your Hugging Face")
    else:
        user_data = api.whoami()
        if (username['auth']['accessToken']['role'] != "write"):
            gr.Warning("Oops, you've uploaded a `Read` token. You need to use a Write token!")
        else:
            if user_data['canPay']:
                return gr.update(visible=False), gr.update(visible=True)    
            else:
                return gr.update(visible=True), gr.update(visible=False)
                
        return gr.update(visible=False), gr.update(visible=False)

with gr.Blocks() as demo:
    dataset_folder = gr.State()
    gr.Markdown("# SDXL LoRA Dreambooth Training")
    lora_name = gr.Textbox(label="The name of your LoRA", placeholder="e.g.: Persian Miniature Painting style, Cat Toy")
    training_option = gr.Radio(
        label="What are you training?", choices=["object", "style", "face", "custom"]
    )
    concept_sentence = gr.Textbox(
        label="Concept sentence",
        info="A common sentence to be used in all images as your captioning structure. TOK is a special mandatory token that will be used to teach the model your concept.",
        placeholder="e.g.: A photo of TOK, in the style of TOK",
        visible=False,
        interactive=True,
    )
    with gr.Group(visible=False) as image_upload:
        with gr.Row():
            images = gr.File(
                file_types=["image"],
                label="Upload your images",
                file_count="multiple",
                interactive=True,
                visible=True,
                scale=1,
            )
            with gr.Column(scale=3, visible=False) as captioning_area:
                with gr.Column():
                    gr.Markdown(
                        """# Custom captioning
To improve the quality of your outputs, you can add a custom caption for each image, describing exactly what is taking place in each of them. Including TOK is mandatory. You can leave things as is if you don't want to include captioning.
                                """
                    )
                    do_captioning = gr.Button("Add AI captions with BLIP-2")
                    output_components = [captioning_area]
                    caption_list = []
                    for i in range(1, MAX_IMAGES + 1):
                        locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                        with locals()[f"captioning_row_{i}"]:
                            locals()[f"image_{i}"] = gr.Image(
                                width=64,
                                height=64,
                                min_width=64,
                                interactive=False,
                                scale=1,
                                show_label=False,
                            )
                            locals()[f"caption_{i}"] = gr.Textbox(
                                label=f"Caption {i}", scale=4
                            )

                        output_components.append(locals()[f"captioning_row_{i}"])
                        output_components.append(locals()[f"image_{i}"])
                        output_components.append(locals()[f"caption_{i}"])
                        caption_list.append(locals()[f"caption_{i}"])
    with gr.Accordion(open=False, label="Advanced options", visible=False) as advanced:
        with gr.Row():
            with gr.Column():
                optimizer = gr.Dropdown(
                    label="Optimizer",
                    info="Prodigy is an auto-optimizer and works good by default. If you prefer to set your own learning rates, change it to AdamW. If you don't have enough VRAM to train with AdamW, pick 8-bit Adam.",
                    choices=[
                        ("Prodigy", "prodigy"),
                        ("AdamW", "adamW"),
                        ("8-bit Adam", "8bitadam"),
                    ],
                    value="prodigy",
                    interactive=True,
                )
                use_snr_gamma = gr.Checkbox(label="Use SNR Gamma")
                snr_gamma = gr.Number(
                    label="snr_gamma",
                    info="SNR weighting gamma to re-balance the loss",
                    value=5.000,
                    step=0.1,
                    visible=False,
                )
                mixed_precision = gr.Dropdown(
                    label="Mixed Precision",
                    choices=["no", "fp16", "bf16"],
                    value="bf16",
                )
                learning_rate = gr.Number(
                    label="UNet Learning rate",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.0000001,
                    value=1.0,  # For prodigy you start high and it will optimize down
                )
                train_batch_size = gr.Number(label="Train batch size", value=2)
                max_train_steps = gr.Number(
                    label="Max train steps", minimum=1, maximum=50000, value=1000
                )
                lora_rank = gr.Number(
                    label="LoRA Rank",
                    info="Rank for the Low Rank Adaptation (LoRA), a higher rank produces a larger LoRA",
                    value=8,
                    step=2,
                    minimum=2,
                    maximum=1024,
                )
                repeats = gr.Number(
                    label="Repeats",
                    info="How many times to repeat the training data.",
                    value=1,
                    minimum=1,
                    maximum=200,
                )
            with gr.Column():
                with_prior_preservation = gr.Checkbox(
                    label="Prior preservation loss",
                    info="Prior preservation helps to ground the model to things that are similar to your concept. Good for faces.",
                    value=False,
                )
                with gr.Column(visible=False) as prior_preservation_params:
                    with gr.Tab("prompt"):
                        class_prompt = gr.Textbox(
                            label="Class Prompt",
                            info="The prompt that will be used to generate your class images",
                        )

                    with gr.Tab("images"):
                        class_images = gr.File(
                            file_types=["image"],
                            label="Upload your images",
                            file_count="multiple",
                        )
                    num_class_images = gr.Number(
                        label="Number of class images, if there are less images uploaded then the number you put here, additional images will be sampled with Class Prompt",
                        value=20,
                    )
                train_text_encoder_ti = gr.Checkbox(
                    label="Do textual inversion",
                    value=True,
                    info="Will train a textual inversion embedding together with the LoRA. Increases quality significantly.",
                )
                with gr.Group(visible=True) as pivotal_tuning_params:
                    train_text_encoder_ti_frac = gr.Number(
                        label="Pivot Textual Inversion",
                        info="% of epochs to train textual inversion for",
                        value=0.5,
                        step=0.1,
                    )
                    num_new_tokens_per_abstraction = gr.Number(
                        label="Tokens to train",
                        info="Number of tokens to train in the textual inversion",
                        value=2,
                        minimum=1,
                        maximum=1024,
                        interactive=True,
                    )
                with gr.Group(visible=False) as text_encoder_train_params:
                    train_text_encoder = gr.Checkbox(
                        label="Train Text Encoder", value=True
                    )
                    train_text_encoder_frac = gr.Number(
                        label="Pivot Text Encoder",
                        info="% of epochs to train the text encoder for",
                        value=0.8,
                        step=0.1,
                    )
                text_encoder_learning_rate = gr.Number(
                    label="Text encoder learning rate",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.0000001,
                    value=1.0,
                )
                seed = gr.Number(label="Seed", value=42)
                resolution = gr.Number(
                    label="Resolution",
                    info="Only square sizes are supported for now, the value will be width and height",
                    value=1024,
                )

        with gr.Accordion(open=False, label="Even more advanced options"):
            with gr.Row():
                with gr.Column():
                    num_train_epochs = gr.Number(label="num_train_epochs", value=1)
                    checkpointing_steps = gr.Number(
                        label="checkpointing_steps", value=5000
                    )
                    prior_loss_weight = gr.Number(label="prior_loss_weight", value=1)
                    gradient_accumulation_steps = gr.Number(
                        label="gradient_accumulation_steps", value=1
                    )
                    gradient_checkpointing = gr.Checkbox(
                        label="gradient_checkpointing",
                        info="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass",
                        value=True,
                    )
                    enable_xformers_memory_efficient_attention = gr.Checkbox(
                        label="enable_xformers_memory_efficient_attention"
                    )
                    adam_beta1 = gr.Number(
                        label="adam_beta1", value=0.9, minimum=0, maximum=1, step=0.01
                    )
                    adam_beta2 = gr.Number(
                        label="adam_beta2", minimum=0, maximum=1, step=0.01, value=0.99
                    )
                    prodigy_beta3 = gr.Number(
                        label="Prodigy Beta 3",
                        value=None,
                        step=0.01,
                        minimum=0,
                        maximum=1,
                    )
                    prodigy_decouple = gr.Checkbox(label="Prodigy Decouple")
                    adam_weight_decay = gr.Number(
                        label="Adam Weight Decay",
                        value=1e-04,
                        step=0.00001,
                        minimum=0,
                        maximum=1,
                    )
                    adam_weight_decay_text_encoder = gr.Number(
                        label="Adam Weight Decay Text Encoder",
                        value=None,
                        step=0.00001,
                        minimum=0,
                        maximum=1,
                    )
                    adam_epsilon = gr.Number(
                        label="Adam Epsilon",
                        value=1e-08,
                        step=0.00000001,
                        minimum=0,
                        maximum=1,
                    )
                    prodigy_use_bias_correction = gr.Checkbox(
                        label="Prodigy Use Bias Correction", value=True
                    )
                    prodigy_safeguard_warmup = gr.Checkbox(
                        label="Prodigy Safeguard Warmup", value=True
                    )
                    max_grad_norm = gr.Number(
                        label="Max Grad Norm",
                        value=1.0,
                        minimum=0.1,
                        maximum=10,
                        step=0.1,
                    )
                with gr.Column():
                    scale_lr = gr.Checkbox(
                        label="Scale learning rate",
                        info="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size",
                    )
                    lr_num_cycles = gr.Number(label="lr_num_cycles", value=1)
                    lr_scheduler = gr.Dropdown(
                        label="lr_scheduler",
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ],
                        value="constant",
                    )
                    lr_power = gr.Number(
                        label="lr_power", value=1.0, minimum=0.1, maximum=10
                    )
                    lr_warmup_steps = gr.Number(label="lr_warmup_steps", value=0)
                    dataloader_num_workers = gr.Number(
                        label="Dataloader num workers", value=0, minimum=0, maximum=64
                    )
                    local_rank = gr.Number(label="local_rank", value=-1)
    with gr.Row(visible=False) as cost_estimation:
        with gr.Group():
            gr.Markdown('''### This training is estimated to cost <b>< US$ 1,50</b> with your current train settings
Grab a Hugging Face <b>write</b> token [here](https://huggingface.co/settings/tokens) 
            ''')
        token = gr.Textbox(label="Your Hugging Face write token", info="A Hugging Face write token you can obtain on the settings page")
    with gr.Group(visible=False) as no_payment_method:
        with gr.Row():
            gr.Markdown("Your Hugging Face account doesn't have a payment method. Set it up [here](https://huggingface.co/settings/billing/payment) to train your LoRA")
            payment_setup = gr.Button("I have set up my payment method")
    start = gr.Button("Start training", visible=False)
    progress_area = gr.HTML("")
    output_components.insert(1, advanced)
    output_components.insert(1, cost_estimation)
    
    gr.on(
        triggers=[
            token.change,
            payment_setup.click
        ],
        fn=check_token,
        inputs=token,
        outputs=[no_payment_method, start]
    )
    use_snr_gamma.change(
        lambda x: gr.update(visible=x),
        inputs=use_snr_gamma,
        outputs=snr_gamma,
        queue=False,
    )
    with_prior_preservation.change(
        lambda x: gr.update(visible=x),
        inputs=with_prior_preservation,
        outputs=prior_preservation_params,
        queue=False,
    )
    train_text_encoder_ti.change(
        lambda x: gr.update(visible=x),
        inputs=train_text_encoder_ti,
        outputs=pivotal_tuning_params,
        queue=False,
    ).then(
        lambda x: gr.update(visible=(not x)),
        inputs=train_text_encoder_ti,
        outputs=text_encoder_train_params,
        queue=False,
    )
    train_text_encoder.change(
        lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=train_text_encoder,
        outputs=[train_text_encoder_frac, text_encoder_learning_rate],
        queue=False,
    )
    class_images.change(
        lambda x: gr.update(value=len(x)),
        inputs=class_images,
        outputs=num_class_images,
        queue=False
    )
    images.upload(
        load_captioning, inputs=[images, concept_sentence], outputs=output_components
    ).then(
        change_defaults,
        inputs=[training_option, images],
        outputs=[max_train_steps, repeats, lr_scheduler, lora_rank, with_prior_preservation, class_prompt, class_images]
    )
    images.change(
        check_removed_and_restart,
        inputs=[images],
        outputs=[captioning_area, advanced, cost_estimation],
    )
    training_option.change(
        make_options_visible,
        inputs=training_option,
        outputs=[concept_sentence, image_upload],
    )
    start.click(
        fn=create_dataset,
        inputs=[images] + caption_list,
        outputs=dataset_folder
    ).then(
        fn=start_training,
        inputs=[
            lora_name,
            training_option,
            concept_sentence,
            optimizer,
            use_snr_gamma,
            snr_gamma,
            mixed_precision,
            learning_rate,
            train_batch_size,
            max_train_steps,
            lora_rank,
            repeats,
            with_prior_preservation,
            class_prompt,
            class_images,
            num_class_images,
            train_text_encoder_ti,
            train_text_encoder_ti_frac,
            num_new_tokens_per_abstraction,
            train_text_encoder,
            train_text_encoder_frac,
            text_encoder_learning_rate,
            seed,
            resolution,
            num_train_epochs,
            checkpointing_steps,
            prior_loss_weight,
            gradient_accumulation_steps,
            gradient_checkpointing,
            enable_xformers_memory_efficient_attention,
            adam_beta1,
            adam_beta2,
            prodigy_beta3,
            prodigy_decouple,
            adam_weight_decay,
            adam_weight_decay_text_encoder,
            adam_epsilon,
            prodigy_use_bias_correction,
            prodigy_safeguard_warmup,
            max_grad_norm,
            scale_lr,
            lr_num_cycles,
            lr_scheduler,
            lr_power,
            lr_warmup_steps,
            dataloader_num_workers,
            local_rank,
            dataset_folder,
            token
        ],
        outputs = progress_area
    )

    do_captioning.click(
        fn=run_captioning, inputs=[images] + caption_list + [training_option], outputs=caption_list
    )
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)