import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def generate_images(
    nationalities, model_name="runwayml/stable-diffusion-v1-5", out_dir="./working"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for nat in nationalities:
        prompt = (
            f"Portrait of a {nat} person, highly detailed, professional photography"
        )
        with torch.autocast(device):
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
        filename = os.path.join(out_dir, f"{nat.replace(' ', '_')}.png")
        image.save(filename)
        paths[nat] = filename
    return paths


def evaluate_generated(paths):
    success = 0
    for nat, path in paths.items():
        try:
            with Image.open(path) as img:
                img.verify()
            success += 1
        except Exception:
            pass
    score = success / len(paths)
    return score


if __name__ == "__main__":
    # Test nationalities
    test_nationalities = ["Afghans", "Americans", "Nigerians"]
    generated_paths = generate_images(test_nationalities)
    score = evaluate_generated(generated_paths)
    print(f"Generation success rate: {score:.2f}")
