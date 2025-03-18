import os
import requests
import PIL
from io import BytesIO

import torch

from diffusers import FluxPipeline
from diffusers.training_utils import set_seed
from scheduling_flow_match_euler_discrete_sde import FlowMatchEulerDiscreteSDEScheduler
from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

example_image = download_image("https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")


def test_flux(enable_sde: bool = False):
    set_seed(999)
    if enable_sde:
        orig_scheduler = pipe.scheduler
        scheduler=FlowMatchEulerDiscreteSDEScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
            subfolder="scheduler",
        )
        pipe.scheduler = scheduler

    edited_image = pipe(prompt="a tomato", num_inference_steps=28).images[0]

    save_dir = "./results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_save_path = os.path.join(save_dir, f"flux_{'sde' if enable_sde else 'ode'}_sampling.png")
    edited_image.save(image_save_path)
    print('Results saved here: ', image_save_path)

    if enable_sde:
        # restore the original scheduler
        pipe.scheduler = orig_scheduler


def test_rf_inversion_sde_sampling(enable_sde: bool = False):
    set_seed(999)
    pipe_rf_inversion = RFInversionFluxPipelineSDE.from_pipe(pipe)

    inverted_latents, image_latents, latent_image_ids = pipe_rf_inversion.invert(
        image=example_image, 
        num_inversion_steps=28, 
        gamma=0.5
    )

    edited_image = pipe_rf_inversion(
        prompt="a tomato",
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        start_timestep=0,
        stop_timestep=7/28,
        num_inference_steps=28,
        eta=0.9,    
        enable_sde=enable_sde,
    ).images[0]

    save_dir = "./results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_save_path = os.path.join(save_dir, f"rf_inversion_{'sde' if enable_sde else 'ode'}_sampling.png")
    edited_image.save(image_save_path)
    print('Results saved here: ', image_save_path)


if __name__ == "__main__":
    test_rf_inversion_sde_sampling(enable_sde=True)
    test_rf_inversion_sde_sampling(enable_sde=False)
    test_flux(enable_sde=True)
    test_flux(enable_sde=False)