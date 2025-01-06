import os
import modal
from pathlib import Path
import torch
from peft import PeftModel
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from components.config import DEFAULT_CONFIG
from components.formatters import format_for_tokenization
from components.models import setup_tokenizer, load_base_model, load_adapter

VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

# Modal setup
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate", "transformers", "torch", "datasets",
        "tensorboard", "trl==0.12.2", "peft", "bitsandbytes",
        "wheel", "tensorflow", "h5py"
    ).run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="llama3.1-70b28b-cot-distillation-wsdm", image=image)

def generate_logits_for_batch(model, sequences, max_seq_len, tokenizer):
    inputs = tokenizer(
        sequences["text"], 
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        padding=False,
        max_length=max_seq_len,
        return_overflowing_tokens=False,
        return_length=False,
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    if len(inputs["input_ids"][0]) > max_seq_len:
        logits = torch.zeros((len(inputs["input_ids"]), max_seq_len, model.config.vocab_size), dtype=torch.float16)
    else: 
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
    
    return logits

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def gen_logits(dataset, config=None):
    if config is None:
        config = DEFAULT_CONFIG
        
    dataset = Dataset.from_list(dataset)
    
    # Setup tokenizer and model
    tokenizer = setup_tokenizer(config["models"]["teacher"], config)
    model = load_base_model(config["models"]["teacher"], config)
    
    if config["models"]["teacher_adapter"]:
        model = load_adapter(model, config["models"]["teacher_adapter"])

    batch_size = 1
    max_seq_len = config["tokenizer"]["max_length"]
    file_name = os.path.join(VOL_MOUNT_PATH, "lora_model/distillation_logits-1.tfrecord")
    
    try:
        with tf.io.TFRecordWriter(file_name) as writer:
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:i+batch_size]
                logits = generate_logits_for_batch(model, batch, max_seq_len, tokenizer)
                
                logits = logits.half().cpu().numpy()
                actual_seq_len = logits.shape[1]
                
                if actual_seq_len > max_seq_len:
                    logits = logits[:, :max_seq_len, :]
                
                feature = {
                    'logits': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[logits.tobytes()]
                    )),
                    'seq_len': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[logits.shape[1]]
                    ))
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    finally:
        output_vol.commit()

def main():
    dataset = load_from_disk("/Users/agokrani/Documents/experiments/aideml/wsdm/wsdm2024-cot-dataset/shard_0")
    dataset = dataset.map(
        format_for_tokenization(tokenizer),
        batched=True,
    )
    select_indices = list(range(100, 200))
    
    with app.run():    
        gen_logits.remote(dataset.select(select_indices).to_list())

if __name__ == "__main__":
    main()
import os
import modal
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
from datasets import load_from_disk, Dataset

from components.config import DEFAULT_CONFIG
from components.models import setup_tokenizer, load_models
from components.formatters import format_for_tokenization

# Modal setup
VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate", "transformers", "torch", "datasets",
        "tensorboard", "trl==0.12.2", "peft", "bitsandbytes",
        "wheel", "tensorflow", "h5py"
    ).run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="generate-logits", image=image)

def generate_logits_for_batch(model, sequences, max_seq_len, tokenizer):
    inputs = tokenizer(
        sequences["text"], 
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        padding=False,
        max_length=max_seq_len,
        return_overflowing_tokens=False,
        return_length=False,
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    return logits

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def gen_logits(dataset, config=None):
    if config is None:
        config = DEFAULT_CONFIG
        
    dataset = Dataset.from_list(dataset)
    
    # Load models and tokenizer using components
    models = load_models(config)
    teacher_model = models.get("teacher_model")
    tokenizer = models["student_tokenizer"]  # We can use either tokenizer
    
    if teacher_model is None:
        raise ValueError("Teacher model is required for generating logits")

    batch_size = 1
    max_seq_len = config["tokenizer"]["max_length"]
    file_name = os.path.join(VOL_MOUNT_PATH, "logits/distillation_logits.tfrecord")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    try:
        with tf.io.TFRecordWriter(file_name) as writer:
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:i+batch_size]
                logits = generate_logits_for_batch(teacher_model, batch, max_seq_len, tokenizer)
                
                logits = logits.half().cpu().numpy()
                
                feature = {
                    'logits': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[logits.tobytes()]
                    )),
                    'seq_len': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[logits.shape[1]]
                    ))
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    finally:
        output_vol.commit()

def main():
    config = DEFAULT_CONFIG
    config.update({
        "models": {
            "teacher": "meta-llama/Llama-3.1-70B-Instruct",
            "student": "meta-llama/Llama-3.1-8B-Instruct",
        }
    })
    
    dataset = load_from_disk("/path/to/dataset")
    tokenizer = setup_tokenizer(config["models"]["teacher"], config)
    
    dataset = dataset.map(
        format_for_tokenization(tokenizer),
        batched=True,
    )
    
    with app.run():    
        gen_logits.remote(dataset.select(range(100)).to_list(), config)

if __name__ == "__main__":
    main()
