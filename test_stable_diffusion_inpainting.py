import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter
import os

# Load the model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,  # Use FP16 for efficiency
    use_safetensors=False       # Use .bin files
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available, else CPU
pipe.enable_attention_slicing()  # Optimize for low VRAM

# Function to load images
def load_image(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at {path}")
        image = Image.open(path).convert("RGB").resize((512, 512))
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

# Local image paths
init_image_path = "images/lady_on_the_street_2.jpg"
mask_image_path = "images/Screenshot 2025-06-03 131612.png"

# Load images
init_image = load_image(init_image_path)
mask_image = load_image(mask_image_path).convert("L")  # Grayscale mask

# Apply Gaussian blur to mask
mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=12))

# Check mask values
print(f"Mask min: {mask_image.getextrema()[0]}, max: {mask_image.getextrema()[1]}")

# Refined prompt
prompt = "A realistic man standing, wearing a black suit, short black hair, facing forward, matching lighting with the girl."

# Perform inpainting
results = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    strength=1.0,  # Full generation
    guidance_scale=12,  # CFG Scale
    num_inference_steps=50,  # Steps for quality
    num_images_per_prompt=4  # Batch size
).images

# Save results
for i, result in enumerate(results):
    result.save(f"output_image_{i}.png")
    result.show()  # Optional: display image
    
    
