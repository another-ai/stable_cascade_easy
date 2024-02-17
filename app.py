import os
import sys
import gc
path = os.path.abspath("src")
sys.path.append(path)
import time
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline # Stable Cascade
from diffusers import LCMScheduler # LCM Scheduler
# from diffusers import DPMSolverSinglestepScheduler # Euler a Scheduler error
from diffusers import DPMSolverMultistepScheduler # DPM++ 2M Karras Scheduler
# from diffusers import EulerAncestralDiscreteScheduler  # DPM++ SDE Karras Scheduler error
import gradio as gr
import random
from PIL import ImageEnhance
import image_save_file
from dotenv import load_dotenv

def constrast_image(image_file, factor):
    im_constrast = ImageEnhance.Contrast(image_file).enhance(factor)
    return im_constrast

def image_print_create(prompt,negative_prompt,sampler_choice,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decode,contrast):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    num_images_per_prompt = 1

    if prompt =="":
        prompt = "a cat with the sign: prompt not found, write in black"
    negative_prompt = negative_prompt

    if random_seed:
        input_seed = random.randint(0, 9999999999)
    else:
        input_seed = int(input_seed)

    if float(guidance_scale).is_integer():
        guidance_scale = int(guidance_scale) # for txt_file_data correct format

    print("Prompt: " + prompt)
        
    resize_pixel_w = width % 128
    resize_pixel_h = height % 128
    if resize_pixel_w > 0:
        width = width - resize_pixel_w
    if resize_pixel_h > 0:
        height = height - resize_pixel_h

    generator = torch.Generator(device=device).manual_seed(input_seed)
    
    start_time = time.time()
    
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
    prior.safety_checker = None
    prior.requires_safety_checker = False

    
    match sampler_choice:
        case "DPM++ 2M Karras":
            sampler = "DPM++ 2M Karras"
            prior.scheduler = DPMSolverMultistepScheduler.from_config(prior.scheduler.config, use_karras_sigmas='true')
        case "LCM":
            sampler = "LCM"
            prior.scheduler = LCMScheduler.from_config(prior.scheduler.config) 
        case _:
            sampler = "DDPMWuerstchenScheduler" #default

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

    del prior
    gc.collect()
    if device=="cuda":
        torch.cuda.empty_cache()

    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to(device)
    decoder.safety_checker = None
    decoder.requires_safety_checker = False

    """ # error with different scheduler in decode than DDPMWuerstchenScheduler
    match sampler_choice:
        case "DPM++ 2M Karras":
            decoder.scheduler = DPMSolverMultistepScheduler.from_config(decoder.scheduler.config, use_karras_sigmas='true')
        case "LCM":
            decoder.scheduler = LCMScheduler.from_config(decoder.scheduler.config) 
        case _:
            sampler = "DDPMWuerstchenScheduler" #default
    """

    image = decoder(image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0,
        generator=generator,
        num_inference_steps=num_inference_steps_decode,
        output_type="pil"
    ).images[0]

    end_time = time.time()

    duration = end_time - start_time

    print(f"Time: {duration} seconds.")
    
    if resize_pixel_w > 0:
        width = width + resize_pixel_w
    if resize_pixel_h > 0:
        height = height + resize_pixel_h

    if resize_pixel_w > 0 or resize_pixel_h > 0:
        image = image.resize((width, height))

    if contrast != 1:
        image = constrast_image(image, contrast)

    txt_file_data=prompt+"\n"+"Negative prompt: "+negative_prompt+"\n"+"Steps: "+str(num_inference_steps)+", Sampler: "+sampler+", CFG scale: "+str(guidance_scale)+", Seed: "+str(input_seed)+", Size: "+str(width)+"x"+str(height)+", Model: stable_cascade"

    file_path = image_save_file.save_file(image, txt_file_data)

    del decoder
    gc.collect()
    if device=="cuda":
        torch.cuda.empty_cache()

    return_txt_file_data = f"{txt_file_data}\nTime: {duration} seconds."
    return image, return_txt_file_data

if __name__ == "__main__":

    load_dotenv("./env/.env")

    default_negative_prompt = os.getenv("negative_prompt", "")
    default_sampler = os.getenv("sampler", "DDPMWuerstchenScheduler")
    default_random_seed = os.getenv("random_seed", "true").lower() == "true"
    default_input_seed = int(os.getenv("input_seed", "1234"))
    default_width = int(os.getenv("width", "768"))
    default_height = int(os.getenv("height", "1024"))
    default_guidance_scale = float(os.getenv("guidance_scale", "4"))
    default_num_inference_steps = int(os.getenv("num_inference_steps", "20"))
    default_num_inference_steps_decode = int(os.getenv("num_inference_steps_decode", "12"))
    default_contrast = float(os.getenv("contrast", "1"))
    sampler_choice_list= ["DDPMWuerstchenScheduler","DPM++ 2M Karras","LCM"]
    
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            title="stable_cascade_easy"
            prompt=gr.Textbox(value="", lines=4, label="Prompt")
            negative_prompt=gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt")
            sampler_choice=gr.Dropdown(value=default_sampler, choices=sampler_choice_list, label="Scheduler")
            random_seed=gr.Checkbox(value=default_random_seed, label="Random Seed")
            input_seed=gr.Number(value=default_input_seed, label="Input Seed",step=1,minimum=0, maximum=9999999999)
            width=gr.Number(value=default_width, label="Width",step=100)
            height=gr.Number(value=default_height, label="Height",step=100)
            guidance_scale=gr.Number(value=default_guidance_scale, label="Guidance Scale",step=1)
            num_inference_steps=gr.Number(value=default_num_inference_steps, label="Steps Prior",step=1)
            num_inference_steps_decode=gr.Number(value=default_num_inference_steps_decode, label="Steps Decode",step=1)
            contrast=gr.Slider(value=default_contrast, label="Contrast",step=0.05,minimum=0.5,maximum=1.5)
            btn = gr.Button(value="Submit")
        with gr.Column():
            output_image=gr.Image()
            output_text=gr.Textbox()
    btn.click(image_print_create, inputs=[prompt, negative_prompt,sampler_choice,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decode,contrast],outputs=[output_image,output_text])
demo.launch(inbrowser=True)
