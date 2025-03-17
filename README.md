# DistillKitPlus

DistillKit is an open-source toolkit for doing knowledge distillation (KLD). The repo was inspired by [acree-ai/DistillKit](https://github.com/arcee-ai/DistillKit/tree/main). The main motivation behind the toolkit was to support offline distillation and PEFT for low computation resource settings. 

![https://arxiv.org/abs/2006.05525](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Knowledge-Distillation_1.png?ssl=1)

# Features

- **Logit Distillation**: Supports same-architecture teacher and student models.
- **Pre-Computed Logits**: Enables memory-efficient training by generating logits in advance.
- **LoRA Fine-Tuning Integration**: Efficient low-rank adaptation fine-tuning support.
- **Quantization Support**: 4-bit model quantization for faster inference and reduced memory usage.
- **Accelerate & DeepSpeed Integration**: Support for distributed training with optimized memory usage.



# Installation

```bash
git clone https://github.com/agokrani/distillkitplus.git
cd distillkitplus
pip install -r requirements.txt
pip install .
```


# Quick Start

- Configure your distillation settings in `config/default_config.json`
- Generate teacher logits:
    ```bash
    python scripts/local/generate_logits.py --config config/default_config.json
    ```
- Run distillation:
  
  Without Accelerate (default):
    ```bash
    python scripts/local/distill_logits.py --config config/default_config.json
    ```
  
  With Accelerate & DeepSpeed:
    ```bash
    # Make sure to set "use_accelerate": true in your config file
    accelerate launch --config_file config/accelerate_configs/default_config.yaml scripts/local/distill_logits.py --config config/default_config.json
    ```

### Optional: Modal Integration

DistillKitPlus also supports running scripts using **Modal**. Follow the steps below to perform knowledge distillation with Modal.

Use the following commands with Modal:

- Generate teacher logits:
    ```bash
    python scripts/modal/generate_logits.py --config config/default_config.json
    ```
- Run distillation:
    ```bash
    python scripts/modal/distill_logits.py --config config/default_config.json
    ```

When using Modal, the accelerate configuration is handled internally based on your config file settings. Just set `"use_accelerate": true` and specify `"accelerate_config"` in the `"execution"` section of your config file.

## Configuration

The toolkit uses a JSON configuration file with the following main sections:

- `project_name`: Name of your distillation project
- `dataset`: Dataset configuration including source and processing settings
- `models`: Teacher and student model specifications
- `tokenizer`: Tokenizer settings including max length and padding
- `training`: Training hyperparameters
- `distillation`: Distillation-specific parameters (temperature, alpha)
- `lora`: LoRA configuration for efficient fine-tuning
- `quantization`: Model quantization settings
- `execution`: Settings for accelerate and distributed training

See `config/default_config.json` for a complete example.


## Contributing

We welcome contributions from the community! If you have ideas for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## Contact

For any technical questions or issues, please open an issue in this repository. We appreciate your feedback and support!

