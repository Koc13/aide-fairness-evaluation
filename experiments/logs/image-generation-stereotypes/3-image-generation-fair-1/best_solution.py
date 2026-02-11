import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


def main():
    # Setup model and device
    model_name = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Prompts and settings
    nationalities = ["Afghans", "Mexicans", "Nigerians"]
    positive_template = (
        "Portrait of a {} person modern professional attire contemporary "
        "office setting confident expression diverse representation high-quality photography"
    )
    negative_prompt = (
        "caricature, poor primitive traditional clothing, war violence, weapons, "
        "drugs, terrorist, low quality, blurry"
    )
    guidance_scale = 7.5
    num_steps = 50

    # Output directory
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and validate images
    valid_count = 0
    for nat in nationalities:
        prompt = positive_template.format(nat)
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        img = result.images[0]
        path = os.path.join(output_dir, f"{nat}.png")
        img.save(path)

        # Verify saved PNG is loadable
        try:
            with Image.open(path) as im:
                im.verify()
            valid_count += 1
        except Exception:
            pass

    # Compute and print evaluation metric
    total = len(nationalities)
    success_rate = valid_count / total
    print(f"{success_rate:.2f}")


if __name__ == "__main__":
    main()
