import modal
import argparse
from pathlib import Path
from scripts.local.generate_logits import gen_logits
from components.config import load_config


VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("distillation-volume", create_if_missing=True)

# Modal setup
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "transformers",
        "torch",
        "datasets",
        "tensorboard",
        "trl==0.12.2",
        "peft",
        "bitsandbytes",
        "wheel",
        "tensorflow",
        "h5py",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="generate-logits", image=image)


@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def gen_logits_modal(config):
    try:
        gen_logits(config)
    finally:
        output_vol.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)  # Will load default if args.config is None

    with app.run():
        gen_logits_modal.remote(config)


if __name__ == "__main__":
    main()
