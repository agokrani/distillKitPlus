import os
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from datasets import load_from_disk, load_dataset as hf_load_dataset

from components.config import load_config
from components.formatters import get_formatter
from components.models import setup_tokenizer, load_base_model, load_adapter

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

def load_dataset(config): 
    if config["dataset"]["split"] is None: 
        dataset = load_from_disk(config["dataset"]["name"])
    else:
        if config["dataset"]["name"].endswith((".json", ".jsonl")):
            dataset = hf_load_dataset("json", data_files=config["dataset"]["name"], split=config["dataset"]["split"])
        else: 
            dataset = hf_load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])

    tokenizer = setup_tokenizer(config["models"]["teacher"], config)
    
    # Get format function from config or use default
    format_function = config["dataset"].get("format_function", "default_format")
    format_func = get_formatter(format_function, tokenizer)
    
    dataset = dataset.map(
        format_func,
        batched=True,
    )
    num_samples = config["dataset"]["num_samples"]
    select_range = config["dataset"].get("select_range")
    if num_samples:
        if select_range:
            samples_to_select = list(range(select_range[0], select_range[1]))
            assert num_samples == len(samples_to_select)
            dataset = dataset.select(samples_to_select)
        else: 
            dataset = dataset.select(range(num_samples))
    
    return dataset

def gen_logits(config):
    dataset = load_dataset(config)
    
    config.update({
        "hf_token": os.environ["HF_TOKEN"]
    })
    
    # Setup tokenizer and model
    tokenizer = setup_tokenizer(config["models"]["teacher"], config)
    model = load_base_model(config["models"]["teacher"], config, cache_dir="/vol/base_model", save_base_model_to_cache=True)
    
    if config["models"]["teacher_adapter"]:
        model = load_adapter(model, config["models"]["teacher_adapter"])

    batch_size = 1
    max_seq_len = config["tokenizer"]["max_length"]
    file_name = config["dataset"]["logits_file"]
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/default_config.json", help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    gen_logits(config)

if __name__ == "__main__":
    main()
