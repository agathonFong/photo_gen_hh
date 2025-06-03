import torch
from diffusers import StableDiffusionPipeline

# Adjusted approach to get the model from Hugging Face
## git clone https://<username>:<keycode>@huggingface.co/CompVis/stable-diffusion-v1-4

# Load the Stable Diffusion model without specifying torch_dtype (defaults to torch.float32)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")

# Define a prompt
prompt = "A Chinese Celebrity waiting in front of a KFC"
num_inference_steps = 50
guidance_scale = 7.5

# Generate the image
image = pipe(prompt,
             num_inference_steps=num_inference_steps,
             guidance_scale=guidance_scale
             ).images[0]

# Save the image
image.save("test_image2.png")
print("Image saved as test_image2.png")