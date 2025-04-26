import argparse
from pathlib import Path
import deepspeed
from accelerate import Accelerator
from components.config import load_config
from components.models import load_models
from components.dataset import DistillationDataset
from components.trainer import LogitsTrainer
from components.formatters import get_formatter
from trl import DataCollatorForCompletionOnlyLM, SFTConfig


def train(config):
    use_accelerate = config.get("execution", {}).get("use_accelerate", False)
    accelerator = None

    if use_accelerate:
        accelerator = Accelerator()

    # Load models and tokenizer
    models = load_models(config)
    student_model = models["student_model"]
    student_tokenizer = models["student_tokenizer"]
    teacher_model = models.get("teacher_model")
    teacher_tokenizer = models.get("teacher_tokenizer", None)

    # Ideally teacher vocab size should be provided in the config file. If not we will infer it from teacher model or student model depending on
    # the configuration. Note: If teacher model is not loaded due to logits file. It will be inferred from student model.
    if config["models"]["teacher_vocab_size"] is not None:
        teacher_vocab_size = config["models"]["teacher_vocab_size"]
    elif teacher_model is not None:
        teacher_vocab_size = teacher_model.config.vocab_size
    else:
        if config["distillation"]["loss_type"] == "uld":
            raise ValueError(
                "Please provide teacher vocab size or teacher model in the config file in case of ULD loss."
            )
        teacher_vocab_size = student_model.config.vocab_size

    # Get format function from config or use default
    format_function = config["dataset"].get("format_function", "default_format")
    # Get the formatter function
    format_func = get_formatter(format_function, student_tokenizer)

    # Get teacher formatter if loss type is ULD
    # We need the teacher formatter to format the teacher labels for ULD loss
    # only use ULD loss when doing cross-tokenizer distillation
    teacher_format_func = None
    if config["distillation"]["loss_type"] == "uld":
        teacher_format_func = get_formatter(format_function, teacher_tokenizer)

    assert config["distillation"]["student_response_template"] is not None, (
        "Student response template is not provided in the config file."
    )
    if config["distillation"]["loss_type"] == "uld":
        assert config["distillation"]["teacher_response_template"] is not None, (
            "Teacher response template is not provided in the config file. This is required for ULD loss."
        )

    # Initialize dataset
    dataset = DistillationDataset(
        file_path=config["dataset"]["name"],
        tokenizer=student_tokenizer,
        max_seq_length=config["tokenizer"]["max_length"],
        teacher_vocab_size=teacher_vocab_size,
        teacher_tokenizer=teacher_tokenizer,
        teacher_format_func=teacher_format_func,
        teacher_data_collator=(
            DataCollatorForCompletionOnlyLM(
                config["distillation"]["teacher_response_template"],
                tokenizer=teacher_tokenizer,
            )
            if config["distillation"]["loss_type"] == "uld"
            else None
        ),
        format_func=format_func,
        split=config["dataset"]["split"],
        num_samples=config["dataset"]["num_samples"],
        select_range=config["dataset"].get("select_range"),
        logits_file=config["dataset"]["logits_file"],
        loss_type=config["distillation"]["loss_type"],
    )

    # Training arguments
    training_args = SFTConfig(
        **config["training"],
        max_seq_length=config["tokenizer"]["max_length"],
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
    )

    # Initialize trainer
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=dataset,
        tokenizer=student_tokenizer,
        args=training_args,
        data_collator=DataCollatorForCompletionOnlyLM(
            config["distillation"]["student_response_template"],
            tokenizer=student_tokenizer,
        ),
        # data_collator=DataCollatorForCompletionOnlyLM(
        #     #"<|im_start|>assistant\n",
        #     "<|assistant|>",
        #     tokenizer=student_tokenizer
        # ),
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        loss_type=config["distillation"]["loss_type"],
        k=config["distillation"]["k"],
    )

    if teacher_model is not None:
        if accelerator:
            teacher_model = accelerator.prepare(teacher_model)
            if accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                teacher_model, _, _, _ = deepspeed.initialize(
                    model=teacher_model, config=accelerator.deepspeed_config
                )
        trainer.teacher_model = teacher_model

    if accelerator:
        trainer = accelerator.prepare(trainer)

    # Train and save
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    save_path = Path(config["training"]["output_dir"]) / "final-distilled-checkpoint"
    trainer.save_model(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
