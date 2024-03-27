FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y git gcc g++ && apt-get clean

ENV ROOT=/stable-cascade
RUN --mount=type=cache,target=/root/.cache/pip \
  git clone https://github.com/another-ai/stable_cascade_easy.git ${ROOT}

WORKDIR ${ROOT}

RUN pip install -r requirements.txt 
RUN pip install git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887

RUN sed -i "s/demo.launch(inbrowser=True)/demo.launch(server_name='0.0.0.0',share=False)/" app.py
RUN sed -i "17i torch.backends.cuda.enable_flash_sdp(False)" app.py
RUN sed -i "17i torch.backends.cuda.enable_mem_efficient_sdp(False)" app.py

ENV NVIDIA_VISIBLE_DEVICES=all PYTHONPATH="${PYTHONPATH}:${PWD}" CLI_ARGS=""
EXPOSE 7860
CMD python app.py ${CLI_ARGS}
