import os
import sys
import modal
import subprocess
from pathlib import Path
from components.config import load_config

VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("distillation-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .add_local_dir("scripts", "/root/scripts", copy=True)
    .pip_install(
        "accelerate", "transformers", "torch==2.5.1", "datasets",
        "tensorboard", "trl==0.12.2", "peft", "bitsandbytes",
        "wheel", "tensorflow", "h5py", "tf-keras", "deepspeed"
    ).run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="distill-logits", image=image)

@app.function(
    gpu=modal.gpu.A100(count=8, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_modal(config):
    # Get the path to the local distill_logits.py script
    local_script_path = "scripts/local/distill_logits.py"
    
    # Convert the config object back to a temporary file
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_config_path = f.name
    
    # Check if we should use accelerate
    if config.get("execution", {}).get("use_accelerate", False):
        cmd = ["accelerate", "launch"]
        
        # Add config file if specified
        accelerate_config = config.get("execution", {}).get("accelerate_config", "config/accelerate_configs/default_config.yaml")
        if accelerate_config:
            cmd.extend(["--config_file", accelerate_config])
            
        cmd.extend([local_script_path, "--config", temp_config_path])
    else:
        cmd = [sys.executable, local_script_path, "--config", temp_config_path]
    
    # Execute the command
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Clean up the temporary config file
    os.unlink(temp_config_path)
    
    # Commit the volume changes
    output_vol.commit()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/default_config.json", help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)  # Will load default if args.config is None
    
    with app.run():
        train_modal.remote(config)

if __name__ == "__main__":
    main()
