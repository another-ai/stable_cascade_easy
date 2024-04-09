import os
from datetime import datetime as date_time
import sys
import gc
path = os.path.abspath("src")
sys.path.append(path)
import time
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)
from diffusers import LCMScheduler # LCM Scheduler
import gradio as gr
import random
from PIL import ImageEnhance
import image_save_file
from dotenv import load_dotenv
import platform

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_dtype = torch.bfloat16

def constrast_image(image_file, factor):
    im_constrast = ImageEnhance.Contrast(image_file).enhance(factor)
    return im_constrast

def generate_image(checkpoint_basename,checkpoint_prior,checkpoint_decoder,prompt_input,dynamic_prompt,negative_prompt,sampler_choice,num_images_per_prompt,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decoder,contrast):

    def remove_last_comma(sentence):
        if len(sentence) > 0 and sentence[-1] == ',':
            sentence_without_comma = sentence[:-1]
            return sentence_without_comma
        else:
            return sentence
        
    def remove_duplicates(words):
        words_list = words.split(",")
        unique_words = []
        for word in words_list:
            if word not in unique_words:
                unique_words.append(word)
        unique_string = ",".join(unique_words)
        return unique_string

    if dynamic_prompt > 0:
        if prompt_input != "":
            if prompt_input[-1] != ",":
                prompt_input = prompt_input + ","
        banned_words = os.getenv("banned_words", "").split(",")
        import app_retnet
        prompt = app_retnet.main_def(prompt_input=prompt_input, max_tokens=dynamic_prompt, DEVICE="cpu", banned_words=banned_words, prompt_chara=False)
        prompt = remove_duplicates(prompt.lower())
        prompt = remove_last_comma(prompt)
    else:
        prompt = prompt_input

    if prompt == "":
        prompt = "a cat with the sign: prompt not found, write in black"
    negative_prompt = negative_prompt

    if random_seed:
        input_seed = random.randint(0, 9999999999)
    else:
        input_seed = int(input_seed)

    if float(guidance_scale).is_integer():
        guidance_scale = int(guidance_scale) # for txt_file_data correct format

    generator = torch.Generator(device=device).manual_seed(input_seed)

    print(f"Prompt: {prompt}")

    checkpoint_prior_name = ""
    if len(checkpoint_prior) < 1:
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch_dtype).to(device)
        checkpoint_prior_name = "stable_cascade"
    else:
        prior_unet = StableCascadeUNet.from_single_file(checkpoint_basename + checkpoint_prior,torch_dtype=torch_dtype)
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet, torch_dtype=torch_dtype).to(device)
        checkpoint_prior_name = os.path.splitext(checkpoint_prior)[0]

    prior.safety_checker = None
    prior.requires_safety_checker = False
        
    resize_pixel_w = width % 128
    resize_pixel_h = height % 128
    if resize_pixel_w > 0:
        width = width - resize_pixel_w
    if resize_pixel_h > 0:
        height = height - resize_pixel_h

    start_time = time.time()
    
    match sampler_choice:
        case "LCM":
            sampler = "LCM"
            prior.scheduler = LCMScheduler.from_config(prior.scheduler.config) 
        case _:
            sampler = "DDPMWuerstchenScheduler" # default

    prior_output = prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt
    )

    if len(checkpoint_prior) > 0:
        del prior_unet
    del prior
    gc.collect()
    if device=="cuda":
        torch.cuda.empty_cache()

    # checkpoint_decoder_name = ""
    if len(checkpoint_decoder) < 1:
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch_dtype).to(device)
        # checkpoint_decoder_name = "stable_cascade_decoder"
    else:
        decoder_unet = StableCascadeUNet.from_single_file(checkpoint_basename + checkpoint_decoder,torch_dtype=torch_dtype)
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet, torch_dtype=torch_dtype).to(device)
        # checkpoint_decoder_name = os.path.splitext(checkpoint_decoder)[0]
    decoder.safety_checker = None
    decoder.requires_safety_checker = False

    images = decoder(image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        guidance_scale=0,
        num_inference_steps=num_inference_steps_decoder,
        output_type="pil"
    ).images

    end_time = time.time()

    duration = end_time - start_time

    print(f"Time: {duration} seconds.")
    
    if resize_pixel_w > 0:
        width = width + resize_pixel_w
    if resize_pixel_h > 0:
        height = height + resize_pixel_h

    for image in images:
        if resize_pixel_w > 0 or resize_pixel_h > 0:
            image = image.resize((width, height))

        if contrast != 1:
            image = constrast_image(image, contrast)

        txt_file_data=prompt+"\n"+"Negative prompt: "+negative_prompt+"\n"+"Steps: "+str(num_inference_steps)+", Sampler: "+sampler+", CFG scale: "+str(guidance_scale)+", Seed: "+str(input_seed)+", Size: "+str(width)+"x"+str(height)+", Model: "+checkpoint_prior_name

        file_path = image_save_file.save_file(image, txt_file_data)

    if len(checkpoint_decoder) > 0:
        del decoder_unet
    del decoder
    gc.collect()
    if device=="cuda":
        torch.cuda.empty_cache()

    return_txt_file_data = f"{txt_file_data}\nTime: {duration} seconds."

    yield images, return_txt_file_data

def stop_gen(checkpoint_prior, checkpoint_decoder):
    try:
        if len(checkpoint_prior) > 0:
            del prior_unet
        if len(checkpoint_decoder) > 0:
            del decoder_unet
        del prior
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        os.execv(sys.executable, [sys.executable, __file__, "restart"])    


def open_dir(dir="image"):
    current_datetime = date_time.now()
    current_date = current_datetime.strftime(f"%Y_%m_%d")
    folder = os.getcwd() + "/" + dir + "/" + current_date
    if not os.path.exists(folder):
        folder =  os.getcwd() + "/" + dir

    if os.path.exists(folder):
        operating_system = platform.system()
        if operating_system == 'Windows':
            os.startfile(folder)
        elif operating_system == 'Darwin':
            os.system('open "{}"'.format(folder))
        elif operating_system == 'Linux':
            os.system('xdg-open "{}"'.format(folder))

if __name__ == "__main__":

    load_dotenv("./env/.env")

    default_checkpoint_basename=os.getenv("checkpoint_basename","./stable_cascade/")
    default_negative_prompt = os.getenv("negative_prompt", "")
    default_sampler = os.getenv("sampler", "DDPMWuerstchenScheduler")
    default_batch_size = int(os.getenv("batch_size", "1"))
    default_random_seed = os.getenv("random_seed", "true").lower() == "true"
    default_input_seed = int(os.getenv("input_seed", "1234"))
    default_width = int(os.getenv("width", "768"))
    default_height = int(os.getenv("height", "1024"))
    default_guidance_scale = float(os.getenv("guidance_scale", "4"))
    default_num_inference_steps = int(os.getenv("num_inference_steps", "20"))
    default_num_inference_steps_decoder = int(os.getenv("num_inference_steps_decode", "12"))
    default_contrast = float(os.getenv("contrast", "1"))
    sampler_choice_list= ["DDPMWuerstchenScheduler", "LCM"]
    dynamic_prompt=int(os.getenv("dynamic_prompt", "0"))

    generator_image = generate_image

    inbrowser_ = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "restart":
            inbrowser_ = False 

    
    if default_checkpoint_basename[-1] != "/":
        default_checkpoint_basename = default_checkpoint_basename + "/"

    checkpoints_prior_list = [os.path.basename(file) for file in os.listdir(default_checkpoint_basename) if file.endswith(".safetensors")]
    checkpoints_decoder_list = [os.path.basename(file) for file in os.listdir(default_checkpoint_basename) if file.endswith(".safetensors")]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                title="stable_cascade_easy"
                checkpoint_basename = gr.Textbox(value=default_checkpoint_basename, label="Checkpoint Path", visible=False)
                checkpoint_prior=gr.Dropdown(value=None, choices=checkpoints_prior_list, allow_custom_value=True, filterable=True, label="Checkpoint(Prior, Stage C), empty for Stable Cascade Default Prior")
                checkpoint_decoder=gr.Dropdown(value=None, choices=checkpoints_decoder_list, allow_custom_value=True, filterable=True, label="Checkpoint(Decoder, Stage B), empty for Stable Cascade Default Decoder")
                prompt_input=gr.Textbox(value="", lines=4, label="Prompt")
                dynamic_prompt = gr.Number(value=dynamic_prompt, label="Magic Prompt(max tokens, 0=off)",step=32,minimum=0,maximum=1024)
                negative_prompt=gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt")
                sampler_choice=gr.Dropdown(value=default_sampler, choices=sampler_choice_list, label="Scheduler")
                num_images_per_prompt=gr.Number(value=default_batch_size, label="Batch Size",step=1,minimum=1,maximum=16)
                random_seed=gr.Checkbox(value=default_random_seed, label="Random Seed")
                input_seed=gr.Number(value=default_input_seed, label="Input Seed",step=1,minimum=0, maximum=9999999999)
                width=gr.Number(value=default_width, label="Width",step=100)
                height=gr.Number(value=default_height, label="Height",step=100)
                guidance_scale=gr.Number(value=default_guidance_scale, label="Guidance Scale",step=1)
                with gr.Row():
                    num_inference_steps=gr.Number(value=default_num_inference_steps, label="Steps Prior",step=1)
                    num_inference_steps_decoder=gr.Number(value=default_num_inference_steps_decoder, label="Steps Decoder",step=1)
                contrast=gr.Slider(value=default_contrast, label="Contrast(Default Value = 1)",step=0.05,minimum=0.5,maximum=1.5)
                with gr.Row():  
                    btn_stop_gen = gr.Button(value="Stop")
                    btn_generate = gr.Button(value="Generate")
            with gr.Column():
                output_images=gr.Gallery(allow_preview=True, preview=True, label="Generated Images", show_label=True)
                btn_open_dir = gr.Button(value="Open Image Directory")
                output_text=gr.Textbox(label="Metadata")
        btn_generate.click(generator_image, inputs=[checkpoint_basename, checkpoint_prior,checkpoint_decoder,prompt_input,dynamic_prompt,negative_prompt,sampler_choice,num_images_per_prompt,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decoder,contrast],outputs=[output_images,output_text])
        btn_open_dir.click(open_dir, inputs=[], outputs=[])
        btn_stop_gen.click(stop_gen, inputs=[checkpoint_prior,checkpoint_decoder], outputs=[])
    demo.launch(inbrowser=inbrowser_)
