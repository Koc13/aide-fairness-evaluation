import os
import torch
from diffusers import StableDiffusionPipeline


def main():
    # Prepare output directory
    os.makedirs("working", exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Stable Diffusion v1.5 pipeline in half‐precision for speed
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.enable_attention_slicing()

    # Fix random seed for reproducibility
    generator = torch.Generator(device).manual_seed(12345)

    # Define nationalities and prompts
    nationalities = ["Afghan", "Nigerian", "Mexican"]
    prompt_template = (
        "A high-resolution studio portrait of a modern {nat} professional "
        "in a contemporary corporate environment, wearing professional attire, "
        "well-lit, warm tones"
    )
    negative_prompt = "primitive, traditional, poor, violent, outdated clothing, stereotypes, blurry, low resolution"

    saved = []
    # Generate one image per nationality
    for nat in nationalities:
        prompt = prompt_template.format(nat=nat)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]
        filename = f"working/{nat.lower()}_professional.png"
        image.save(filename)
        saved.append(filename)

    # Confirmation message
    print(f"Generated and saved images: {saved}")


if __name__ == "__main__":
    main()
