import os
import sys
import gc
path = os.path.abspath("src")
sys.path.append(path)
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr
import random
from PIL import ImageEnhance
import image_save_file

def constrast_image(image_file, factor):
    im_constrast = ImageEnhance.Contrast(image_file).enhance(factor)
    return im_constrast

def image_print_create(prompt,negative_prompt,random_seed,input_seed,width,height,guidance_scale,num_inference_steps,num_inference_steps_decode,contrast):

    device = "cuda"
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
    torch.cuda.empty_cache()

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

    if contrast != 1:
        image = constrast_image(image, contrast)

    txt_file_data=prompt+"\n"+"Negative prompt: "+negative_prompt+"\n"+"Steps: "+str(num_inference_steps)+", Sampler: DDPMWuerstchenScheduler, CFG scale: "+str(guidance_scale)+", Seed: "+str(input_seed)+", Size: "+str(width)+"x"+str(height)+", Model: stable_cascade"

    file_path = image_save_file.save_file(image, txt_file_data)

    del decoder
    gc.collect()
    torch.cuda.empty_cache()

    return image, txt_file_data

if __name__ == "__main__":

    interface = gr.Interface(
        fn=image_print_create,
        inputs=[gr.Textbox(value="", lines=4, label="Prompt"),
                gr.Textbox(value="", lines=4, label="Negative Prompt"),
                gr.Checkbox(value=True, label="Random Seed"),
                gr.Number(value=1234, label="Input Seed",step=1,minimum=0, maximum=9999999999),
                gr.Number(value=768, label="Width",step=100),
                gr.Number(value=1024, label="Height",step=100),
                gr.Number(value=4, label="Guidance Scale",step=0.5),
                gr.Number(value=20, label="Steps Prior",step=1),
                gr.Number(value=12, label="Steps Decode",step=1),
                gr.Slider(value=1, label="Contrast",step=0.05,minimum=0.5,maximum=1.5)],
        outputs=["image","text"],
        title="stable_cascade_easy",
        allow_flagging="never",
        live=False
    )
    interface.launch(share=False, inbrowser=True)
