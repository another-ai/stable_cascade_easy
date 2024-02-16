import os
import sys
import gc
path = os.path.abspath("src")
sys.path.append(path)
import time
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr
import random
from PIL import ImageEnhance
import image_save_file
from dotenv import load_dotenv

def constrast_image(image_file, factor):
    im_constrast = ImageEnhance.Contrast(image_file).enhance(factor)
    return im_constrast

def image_print_create(prompt,negative_prompt,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decode,contrast):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    num_images_per_prompt = 1

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
    prior.safety_checker = None
    prior.requires_safety_checker = False

    if prompt =="":
        prompt = "a cat with the sign: prompt not found, write in black"
    negative_prompt = negative_prompt

    if random_seed:
        input_seed = random.randint(0, 9999999999)
    else:
        input_seed = int(input_seed)

    if guidance_scale.is_integer():
        guidance_scale = int(guidance_scale) # for txt_file_data correct format

    generator = torch.Generator(device=device).manual_seed(input_seed)

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

    start_time = time.time()
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to(device)
    decoder.safety_checker = None
    decoder.requires_safety_checker = False

    image = decoder(image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        generator=generator,
        num_inference_steps=num_inference_steps_decode,
        output_type="pil"
    ).images[0]

    end_time = time.time()

    duration = end_time - start_time

    print(f"Time: {duration} seconds.")
    
    if contrast != 1:
        image = constrast_image(image, contrast)

    txt_file_data=prompt+"\n"+"Negative prompt: "+negative_prompt+"\n"+"Steps: "+str(num_inference_steps)+", Sampler: DDPMWuerstchenScheduler, CFG scale: "+str(guidance_scale)+", Seed: "+str(input_seed)+", Size: "+str(width)+"x"+str(height)+", Model: stable_cascade"

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
    default_random_seed = os.getenv("random_seed", "true").lower() == "true"
    default_input_seed = int(os.getenv("input_seed", "1234"))
    default_width = int(os.getenv("width", "768"))
    default_height = int(os.getenv("height", "1024"))
    default_guidance_scale = float(os.getenv("guidance_scale", "4"))
    default_num_inference_steps = int(os.getenv("num_inference_steps", "20"))
    default_num_inference_steps_decode = int(os.getenv("num_inference_steps_decode", "12"))
    default_contrast = float(os.getenv("contrast", "1"))

    interface = gr.Interface(
        fn=image_print_create,
        inputs=[gr.Textbox(value="", lines=4, label="Prompt"),
                gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt"),
                gr.Checkbox(value=default_random_seed, label="Random Seed"),
                gr.Number(value=default_input_seed, label="Input Seed",step=1,minimum=0, maximum=9999999999),
                gr.Number(value=default_width, label="Width",step=100),
                gr.Number(value=default_height, label="Height",step=100),
                gr.Number(value=default_guidance_scale, label="Guidance Scale",step=0.5),
                gr.Number(value=default_num_inference_steps, label="Steps Prior",step=1),
                gr.Number(value=default_num_inference_steps_decode, label="Steps Decode",step=1),
                gr.Slider(value=default_contrast, label="Contrast",step=0.05,minimum=0.5,maximum=1.5)],
        outputs=["image","text"],
        title="stable_cascade_easy",
        allow_flagging="never",
        live=False
    )
    interface.launch(share=False, inbrowser=True)
