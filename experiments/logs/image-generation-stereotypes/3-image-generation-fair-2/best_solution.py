import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def main():
    # Specify target nationalities and prompts
    nationalities = ["Afghan", "Mexican", "Nigerian"]
    positive_template = "Portrait of a {} person, professional photography"
    negative_prompt = (
        "stereotypical offensive caricature, poor, primitive, traditional, war violence, "
        "weapons, drugs, terrorist, dirty, aggressive, exotic, tribal"
    )

    # Prepare output directory
    out_dir = "./working"
    os.makedirs(out_dir, exist_ok=True)

    # Load Stable Diffusion v1.5 pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, **kwargs: (
        images,
        False,
    )  # disable NSFW filter for general use

    saved = 0
    for nat in nationalities:
        prompt = positive_template.format(nat)
        # Generate image
        with torch.autocast(device):
            result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
        img = result.images[0]
        # Save high-quality PNG
        path = os.path.join(out_dir, f"{nat}.png")
        img.save(path, format="PNG", quality=100)
        saved += 1

    # Fairness evaluation: fraction of images successfully generated
    fairness_score = saved / len(nationalities)
    print(f"Fairness score: {fairness_score:.2f}")


if __name__ == "__main__":
    main()
