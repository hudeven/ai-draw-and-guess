import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)  
pipe = pipe.to("cuda")



prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
      image = pipe(prompt)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

      # Now to display an image you can do either save it such as:
      image.save(f"astronaut_rides_horse.png")
