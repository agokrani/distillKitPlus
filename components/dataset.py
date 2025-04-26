import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class TFRecordDataLoader:
    def __init__(self, file_path, vocab_size, valid_indices=None, num_samples=None):
        self.file_path = file_path
        self.feature_description = {
            "logits": tf.io.FixedLenFeature([], tf.string),
            "seq_len": tf.io.FixedLenFeature([], tf.int64),
        }
        self.vocab_size = vocab_size
        self.valid_indices = valid_indices
        self.num_samples = num_samples

        with tf.device("/cpu:0"):
            self.dataset = tf.data.TFRecordDataset(self.file_path)
            self.record_offsets = self._index_tfrecord()

    def _parse_function(self, record):
        parsed_features = tf.io.parse_single_example(record, self.feature_description)
        seq_len = parsed_features["seq_len"].numpy()
        logits_raw = parsed_features["logits"].numpy()
        logits = np.frombuffer(logits_raw, dtype=np.float16)
        logits = logits.reshape((seq_len, self.vocab_size))
        return {"logits": logits, "seq_len": seq_len}

    def _index_tfrecord(self):
        offsets = []
        with tf.device("/cpu:0"):
            if self.num_samples is not None:
                target_samples = self.num_samples
            else:
                target_samples = sum(1 for _ in self.dataset)

            for index, record in enumerate(self.dataset):
                if len(offsets) >= target_samples:
                    break
                if self.valid_indices is None or index in self.valid_indices:
                    offsets.append(index)
        return offsets

    def __len__(self):
        return len(self.record_offsets)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.record_offsets):
                raise IndexError("Index out of range")
            offset = self.record_offsets[idx]
            with tf.device("/cpu:0"):
                for i, record in enumerate(self.dataset):
                    if i == offset:
                        return self._parse_function(record)
            raise IndexError("Index out of range")
        else:
            raise TypeError("Index must be an int")


class DistillationDataset(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer,
        max_seq_length,
        teacher_vocab_size,
        teacher_tokenizer=None,
        teacher_format_func=None,
        teacher_data_collator=None,
        format_func=None,
        split=None,
        num_samples=None,
        logits_file=None,
        select_range=None,
        ignore_index=-100,
        loss_type=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.logits_file = logits_file
        self.ignore_index = ignore_index
        self.split = split
        self.format_func = format_func or self._default_format
        self.teacher_vocab_size = teacher_vocab_size
        self.select_range = select_range
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_format_func = teacher_format_func
        self.teacher_data_collator = teacher_data_collator
        self.loss_type = loss_type

        if self.loss_type == "uld" and self.teacher_tokenizer is None:
            raise ValueError(
                "teacher_tokenizer must be provided when loss_type is 'uld'"
            )

        if self.split is None:
            dataset = load_from_disk(file_path)
        else:
            if file_path.endswith((".json", ".jsonl")):
                dataset = load_dataset("json", data_files=file_path, split=self.split)
            else:
                dataset = load_dataset(file_path, split=self.split)

        self.dataset = dataset.map(
            self.format_func,
            batched=True,
        )
        # hack for to process text with teacher tokenizer
        if self.loss_type == "uld":
            # Create teacher dataset and rename text column to teacher_text
            teacher_dataset = dataset.map(self.teacher_format_func, batched=True)
            teacher_dataset = teacher_dataset.rename_column("text", "teacher_text")

            # Add teacher_text column to main dataset
            self.dataset = self.dataset.add_column(
                "teacher_text", teacher_dataset["teacher_text"]
            )
            # Free up memory by deleting teacher_dataset
            del teacher_dataset

        self.dataset = self.dataset.map(self._tokenize, batched=True)

        if num_samples:
            if self.select_range:
                samples_to_select = list(
                    range(self.select_range[0], self.select_range[1])
                )
                assert num_samples == len(samples_to_select)
                self.dataset = self.dataset.select(samples_to_select)
            else:
                self.dataset = self.dataset.select(range(num_samples))

        self._compute_valid_indices()

        if self.logits_file:
            self.logits = TFRecordDataLoader(
                self.logits_file,
                self.teacher_vocab_size,
                valid_indices=self.valid_indices,
                num_samples=num_samples,
            )
        self.dataset = self.dataset.select(self.valid_indices)

        print(f"dataset length: {len(self.dataset)}")

        if self.logits_file:
            assert hasattr(self, "logits"), "Logits file not found."

            print(f"logits length: {len(self.logits)}")
            assert len(self.dataset) == len(self.logits), (
                "Number of samples in dataset and logits file do not match."
            )

    def _compute_valid_indices(self):
        self.valid_indices = []
        for idx, example in enumerate(self.dataset):
            seq_length = len(example["input_ids"])
            if seq_length <= self.max_seq_length:
                self.valid_indices.append(idx)

    def _tokenize(self, element):
        outputs = self.tokenizer(
            element["text"],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        result = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
        if self.loss_type == "uld" and self.teacher_tokenizer is not None:
            assert "teacher_text" in element, "teacher_text column not found in dataset"
            teacher_outputs = self.teacher_tokenizer(
                element["teacher_text"],
                add_special_tokens=False,
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            batch_size = len(teacher_outputs["input_ids"])

            # Process each example individually through the data collator
            teacher_labels = []
            for i in range(batch_size):
                # Process one example at a time
                single_example = {
                    "input_ids": teacher_outputs["input_ids"][i],
                    "attention_mask": teacher_outputs["attention_mask"][i],
                }
                # Apply data collator to a single example
                processed = self.teacher_data_collator([single_example])
                # Extract labels and add to our list
                example_labels = processed["labels"]
                # If we get back a tensor or list with batch dimension, get the first item
                if hasattr(example_labels, "tolist"):
                    example_labels = example_labels.tolist()
                if isinstance(example_labels, list) and len(example_labels) == 1:
                    example_labels = example_labels[0]
                teacher_labels.append(example_labels)

            result["teacher_labels"] = teacher_labels
            result["teacher_input_ids"] = teacher_outputs["input_ids"]
            result["teacher_attention_mask"] = teacher_outputs["attention_mask"]

        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = {
            "input_ids": self.dataset[index]["input_ids"],
            "attention_mask": self.dataset[index]["attention_mask"],
        }

        if self.loss_type == "uld" and "teacher_input_ids" in self.dataset[index]:
            item["teacher_input_ids"] = self.dataset[index]["teacher_input_ids"]
            item["teacher_attention_mask"] = self.dataset[index][
                "teacher_attention_mask"
            ]
            item["teacher_labels"] = self.dataset[index]["teacher_labels"]

        if self.logits_file:
            item["logits"] = self.logits[index]["logits"]

        return item

    def _default_format(self, examples):
        return {"text": examples["text"] if "text" in examples else examples}
