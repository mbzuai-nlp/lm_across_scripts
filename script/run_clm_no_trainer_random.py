#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=None,
        help="Aligned training files; one representation is selected uniformly per row and epoch.",
    )
    parser.add_argument(
        "--preprocessing_batch_size",
        type=int,
        default=10000,
        help=(
            "Batch size for the preprocessing. Will be passed to the map function of the datasets library to speed up the"
            " preprocessing."
        ),
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the Hugging Face Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="clm_no_trainner",
        help="The name of the project to which the logs will be pushed.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.train_files is None:
        raise ValueError("Need a dataset name, --train_file, or --train_files.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.train_files is not None:
            if len(args.train_files) < 2:
                raise ValueError("--train_files requires at least two aligned files; use --train_file for one file.")
            for train_file in args.train_files:
                extension = train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("Every file in `train_files` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    if args.train_files is not None and args.train_file is not None:
        raise ValueError("Use either --train_file or --train_files, not both.")
    if args.dataset_name is not None and (args.train_file is not None or args.train_files is not None):
        raise ValueError("Use either --dataset_name or local training files, not both.")

    return args

def _load_text_like(
    path_train: str,
    path_val: str | None,
    no_keep_linebreaks: bool,
    validation_split_percentage: int,
):
    data_files = {"train": path_train}
    extension = path_train.split(".")[-1]
    dataset_args = {}

    if path_val is not None:
        if path_val.split(".")[-1] != extension:
            raise ValueError("train/validation extensions must match.")
        data_files["validation"] = path_val

    hf_extension = extension
    if extension == "txt":
        hf_extension = "text"
        dataset_args["keep_linebreaks"] = not no_keep_linebreaks

    ds = load_dataset(hf_extension, data_files=data_files, **dataset_args)

    if "validation" not in ds:
        ds["validation"] = load_dataset(
            hf_extension,
            data_files={"train": path_train},
            split=f"train[:{validation_split_percentage}%]",
            **dataset_args,
        )
        ds["train"] = load_dataset(
            hf_extension,
            data_files={"train": path_train},
            split=f"train[{validation_split_percentage}%:]",
            **dataset_args,
        )

    return ds

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Training seed set to {args.seed}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            os.makedirs(args.output_dir, exist_ok=True)
            gitignore_path = os.path.join(args.output_dir, ".gitignore")
            content = ""
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    content = f.read()
            with open(gitignore_path, "a") as f:
                if content and not content.endswith("\n"):
                    f.write("\n")
                if "step_*" not in content:
                    f.write("step_*\n")
                if "epoch_*" not in content:
                    f.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    multi_file_mode = False
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        if args.train_files is not None:
            views = []
            for train_file in args.train_files:
                logger.info(f"Loading aligned train file: {train_file}")
                views.append(
                    _load_text_like(
                        train_file,
                        args.validation_file,
                        args.no_keep_linebreaks,
                        args.validation_split_percentage,
                    )
                )

            train_size = len(views[0]["train"])
            for train_file, view in zip(args.train_files[1:], views[1:]):
                if len(view["train"]) != train_size:
                    raise ValueError(
                        "All --train_files must align by index and have the same number of rows: "
                        f"expected {train_size}, but {train_file} has {len(view['train'])}."
                    )

            train = views[0]["train"]
            first_column = "text" if "text" in train.column_names else train.column_names[0]
            train = train.rename_column(first_column, "text_0")
            for view_index, view in enumerate(views[1:], start=1):
                text_column = "text" if "text" in view["train"].column_names else view["train"].column_names[0]
                train = train.add_column(f"text_{view_index}", view["train"].data.column(text_column))

            raw_datasets = datasets.DatasetDict(
                {"train": train, "validation": views[0]["validation"]}
            )
            logger.info(f"Created aligned dataset with {len(views)} representations")
            multi_file_mode = True
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
                extension = args.train_file.split(".")[-1]
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
                extension = args.validation_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                    **dataset_args,
                )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    def tokenize_function_views(examples):
        if "text" in examples:
            return tokenizer(examples["text"])

        result = {}
        for view_index in range(len(args.train_files)):
            encoded = tokenizer(examples[f"text_{view_index}"])
            result[f"input_ids_{view_index}"] = encoded["input_ids"]
            result[f"attention_mask_{view_index}"] = encoded["attention_mask"]
        return result

    with accelerator.main_process_first():
        if multi_file_mode:
            tokenized_datasets = raw_datasets.map(
                tokenize_function_views,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Tokenizing aligned training representations",
            )

        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    def choose_input_ids_for_epoch(ds, epoch):
        base_seed = int(args.seed or 0)
        num_views = len(args.train_files)
        uint64_mask = (1 << 64) - 1

        def select_view(row_index):
            # SplitMix64 gives stable, well-mixed choices without depending on
            # process-local random state or Python's hash implementation.
            value = (base_seed + (epoch << 32) + row_index + 0x9E3779B97F4A7C15) & uint64_mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & uint64_mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & uint64_mask
            value ^= value >> 31
            return value % num_views

        def pick_batch(examples, indices):
            selected_views = [select_view(idx) for idx in indices]
            input_ids = [
                examples[f"input_ids_{view_index}"][row_index]
                for row_index, view_index in enumerate(selected_views)
            ]
            attention_mask = [
                examples[f"attention_mask_{view_index}"][row_index]
                for row_index, view_index in enumerate(selected_views)
            ]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "selected_view": selected_views,
            }

        ds_out = ds.map(
            pick_batch,
            batched=True,
            with_indices=True,
            batch_size=args.preprocessing_batch_size,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc=f"Selecting representations for epoch {epoch}",
        )

        counts = [0] * num_views
        for view_index in ds_out["selected_view"]:
            counts[view_index] += 1
        proportions = [count / len(ds_out) for count in counts]
        logger.info(f"[Epoch {epoch}] Representation counts: {counts}")
        logger.info(f"[Epoch {epoch}] Representation proportions: {proportions}")

        return ds_out.remove_columns("selected_view")

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    # Select one aligned representation uniformly for each row and epoch. Selection
    # happens before grouping so chunks reflect that epoch's sampled data.
    def sample_tokenized_datasets(tokenized_datasets, epoch):
        epoch_tok = choose_input_ids_for_epoch(tokenized_datasets["train"], epoch)
        logger.info("=== Detokenizing 10 samples from epoch_tok for debugging ===")

        sample_indices = list(range(min(10, len(epoch_tok))))

        for i in sample_indices:
            ids = epoch_tok[i]["input_ids"]
            text = tokenizer.decode(ids, skip_special_tokens=True)

            logger.info(f"[Sample {i}]: {text}")

        view_columns = [
            column
            for column in epoch_tok.column_names
            if column.startswith(("text_", "input_ids_", "attention_mask_"))
        ]
        epoch_tok_for_group = epoch_tok.remove_columns(view_columns)
        val_ds = tokenized_datasets["validation"]
        if "text" in val_ds.column_names:
            val_ds = val_ds.remove_columns("text")

        return datasets.DatasetDict({"train": epoch_tok_for_group, "validation": val_ds})

    with accelerator.main_process_first():
        if multi_file_mode:
            # this is to compute the total number of updates
            epoch = 0
            first_tokenized_datasets = sample_tokenized_datasets(tokenized_datasets, epoch)

            # Now group_texts on the selected "input_ids"
            with accelerator.main_process_first():
                lm_datasets = first_tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc=f"Grouping texts in chunks of {block_size} (epoch {epoch})",
                )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # show detokenized version for debugging
        ids = train_dataset[index]["input_ids"]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        logger.info(f"Detokenized sample {index} of the training set: {text}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps > 0 else math.ceil(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(args.wandb_project, experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Number of update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Warmup steps = {num_warmup_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    base_tokenized = tokenized_datasets

    for epoch in range(starting_epoch, args.num_train_epochs):
        if multi_file_mode:
            logger.info(
                f"Epoch {epoch}: sampling uniformly from {len(args.train_files)} aligned files: "
                f"{args.train_files}"
            )

            with accelerator.main_process_first():
                epoch_tokenized = sample_tokenized_datasets(base_tokenized, epoch)

                # Now group_texts on the selected "input_ids"
                lm_datasets = epoch_tokenized.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc=f"Grouping texts in chunks of {block_size} (epoch {epoch})",
                )
            train_dataset = lm_datasets["train"]

            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                # show detokenized version for debugging
                ids = train_dataset[index]["input_ids"]
                text = tokenizer.decode(ids, skip_special_tokens=False)
                logger.info(f"Detokenized sample {index} of the training set: {text}.")
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
            )

            # log the new training data size and other stastics after sampling for this epoch
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            logger.info(f"Epoch {epoch}: training dataset size after sampling and grouping: {len(train_dataset)}")
            logger.info(f"Epoch {epoch}: number of update steps per epoch: {num_update_steps_per_epoch}")

            train_dataloader = accelerator.prepare(train_dataloader)

        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

            tokenizer.save_pretrained(output_dir)
            model.config.to_json_file(os.path.join(output_dir, "config.json"))
            with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
                validation_results = {
                    "model_path": output_dir,
                    "perplexity": perplexity,
                    "eval_loss": eval_loss.item(),
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                }
                json.dump(validation_results, f, indent=4)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
