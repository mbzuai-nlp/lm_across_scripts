#!/usr/bin/env python
# coding=utf-8

import argparse
import logging

from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers, trainers
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import Split, Metaspace as PreMetaspace
from tokenizers.decoders import Metaspace as DecoderMetaspace
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    metavar="path",
    type=str,
    nargs="+",
    required=True,
    help="Files to use for training (use quotes for glob patterns)"
)
parser.add_argument("--tokenizer_name", default="new_tokenizer", type=str,
                    help="Output folder for tokenizer")
parser.add_argument("--vocab_size", default=16384, type=int,
                    help="Vocabulary size")
parser.add_argument("--model", default="bpe", type=str,
                    choices=["bpe", "unigram"],
                    help="Type of tokenizer model to train (bpe or unigram)")
args = parser.parse_args()

files = args.files
logger.info(f"Training files: {files}")

# load dataset (text)
dataset = load_dataset("text", data_files={"train": files})
training_corpus = (
    dataset["train"][i : i + 1000]["text"]
    for i in range(0, len(dataset["train"]), 1000)
)

# config
unk_token = "<unk>"
# include reserved boundary token for trainer so it's reserved in the vocab
special_tokens_for_trainer = ["<unk>", "</s>", "<pad>", "▒"]
vocab_size = args.vocab_size

logger.info(f"Training a tokenizer (vocab_size={vocab_size})")

# Tokenizer object
if args.model == "bpe":
    tokenizer = Tokenizer(BPE())
elif args.model == "unigram":
    tokenizer = Tokenizer(Unigram())
else:
    raise ValueError(f"Unsupported tokenizer model: {args.model}")

# Normalizer (keep NFKC)
tokenizer.normalizer = NFKC()

# Pre-tokenizer: force-split on bunsetsu marker then apply Metaspace (pre)
split_bunsetsu = Split("▒", behavior="isolated")
meta_space = PreMetaspace(
    prepend_scheme = "never"
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([split_bunsetsu, meta_space])

# Decoder (decoding uses decoder Metaspace)
tokenizer.decoder = DecoderMetaspace(
    prepend_scheme = "never"
)

# Trainer: ensure '▒' is present in the initial alphabet for stable training
if args.model == "bpe":
    trainer = trainers.BpeTrainer(
        special_tokens=special_tokens_for_trainer,
        vocab_size=vocab_size,
        initial_alphabet=["▒"],
        show_progress=True
    )
elif args.model == "unigram":
    trainer = trainers.UnigramTrainer(
        special_tokens=special_tokens_for_trainer,
        vocab_size=vocab_size,
        show_progress=True
    )

# Train
tokenizer.train_from_iterator(iterator=training_corpus, trainer=trainer)

# Wrap with HF tokenizer
wrapped = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_input_names=["input_ids", "attention_mask"])

# Register named special tokens (this sets their ids)
# If you want single-token-as-boundary (GPT-2-style), map BOS and EOS to same string
wrapped.add_special_tokens({
    "unk_token": "<unk>",
    "bos_token": "</s>",   # using single-boundary token for BOS
    "eos_token": "</s>",   # and EOS (same as GPT-2 style)
    "pad_token": "<pad>",
    "additional_special_tokens": ["▒"],  # bunsetsu marker
})

# Now we can safely get the integer id for the boundary token and set post-processor
vocab = wrapped.get_vocab()
boundary_id = vocab.get("</s>")

# Keep your original GPT-2-like convention (prepend sentinel).
# If you prefer to append EOS instead, change single to "$A </s>" and pair accordingly.
tokenizer.post_processor = TemplateProcessing(
    single="</s> $A",
    pair="</s> $A </s> $B:1",
    special_tokens=[("</s>", boundary_id)],
)

# Save
wrapped.save_pretrained(args.tokenizer_name)
logger.info(f"Tokenizer saved at {args.tokenizer_name}")

# Quick local sanity-check prints (optional)
logger.info("Sanity checks:")
tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_name)
logger.info("tokenize('▒今日は▒ ') -> %s", tok.tokenize("▒今日は▒ "))
ids = tok("▒今日は▒ ")["input_ids"]
logger.info("ids -> %s", ids)
logger.info("with special tokens -> %s", tok.build_inputs_with_special_tokens(ids))
