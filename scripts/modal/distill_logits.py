import os
import sys
import modal
import subprocess
import json
import click
from pathlib import Path
from components.config import load_config

VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("distillation-volume", create_if_missing=True)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "transformers==4.49.0",
        "torch==2.5.1",
        "datasets",
        "tensorboard",
        "trl==0.15.2",
        "bitsandbytes",
        "wheel",
        "tensorflow",
        "h5py",
        "tf-keras",
        "deepspeed==0.16.4",
        "huggingface_hub[hf_transfer]",
    )
    .pip_install(
        "git+https://github.com/agokrani/peft.git@fix-for-modules-to-save-for-zero3#egg=peft"
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .pip_install("click")
    .add_local_dir("scripts", "/root/scripts", copy=True)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(name="distill-logits", image=image)


@app.function(
    gpu="A100-80GB:8",
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_modal(config_str: str):
    # Parse the JSON string back into a dictionary
    config = json.loads(config_str)

    # Get the path to the local distill_logits.py script
    local_script_path = "scripts/local/distill_logits.py"

    # Convert the config object back to a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config_path = f.name

    # Check if we should use accelerate
    if config.get("execution", {}).get("use_accelerate", False):
        cmd = ["accelerate", "launch"]

        # Add config file if specified
        accelerate_config = config.get("execution", {}).get(
            "accelerate_config", "config/accelerate_configs/default_config.yaml"
        )
        if accelerate_config:
            cmd.extend(["--config_file", accelerate_config])

        cmd.extend([local_script_path, "--config", temp_config_path])
    else:
        cmd = [sys.executable, local_script_path, "--config", temp_config_path]

    # Execute the command
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Clean up the temporary config file
    os.unlink(temp_config_path)

    # Commit the volume changes
    output_vol.commit()


@app.local_entrypoint()
@click.option(
    "--config",
    default="config/default_config.json",
    help="Path to config file",
)
def main(config: str):
    # Load config using the path provided by click
    loaded_config = load_config(config)

    # Serialize config to JSON string
    config_str = json.dumps(loaded_config)

    # Call the remote function directly
    train_modal.remote(config_str)
