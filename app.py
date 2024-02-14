import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gc
import gradio as gr
import random

def image_print_create(prompt,negative_prompt,random_seed,input_seed,width,height,guidance_scale,num_inference_steps):

    device = "cuda"
    num_images_per_prompt = 1

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)

    prior.safety_checker = None
    prior.requires_safety_checker = False

    if prompt =="":
        prompt = "a cat"
    negative_prompt = negative_prompt

    if random_seed:
        input_seed = random.randint(0, 9999999999)
    else:
        input_seed = int(input_seed)

    generator = torch.Generator(device=device).manual_seed(input_seed)

    prior_output = prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator = generator,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt
    )

    del prior
    gc.collect()
    torch.cuda.empty_cache()

    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to(device)
    decoder.safety_checker = None
    decoder.requires_safety_checker = False

    image = decoder(image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=12
    ).images[0]

    del decoder
    gc.collect()
    torch.cuda.empty_cache()

    return image

if __name__ == "__main__":

    interface = gr.Interface(
        fn=image_print_create,
        inputs=[gr.Textbox(value="", lines=4, label="Prompt"),
                gr.Textbox(value="", lines=4, label="Negative Prompt"),
                gr.Checkbox(value=True, label="Random Seed"),
                gr.Number(value=1234, label="Input Seed"),
                gr.Number(value=768, label="Width",step=100),
                gr.Number(value=1024, label="Height",step=100),
                gr.Number(value=4, label="Guidance Scale",step=0.5),
                gr.Number(value=20, label="Steps",step=1)],
        outputs="image",
        title="stable_cascade_easy",
        allow_flagging="never",
        live=False
    )
    interface.launch(share=False, inbrowser=True)
