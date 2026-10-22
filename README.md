# Language Modeling Across Script Variants: The Case of Japanese

Data, code, and experiment results for "Language Modeling Across Script Variants: The Case of Japanese."

## Dataset

The preprocessed versions of [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) used in our experiments are hosted on Hugging Face:

**[go-inoue/reazonspeech-script-variants-100m](https://huggingface.co/datasets/go-inoue/reazonspeech-script-variants-100m)**

The dataset contains 17 configurations: `R`, `F0`–`F7`, and `K0`–`K7`. The preprocessing script produces `K` variants through kana conversion and `F` variants through ruby annotations, with levels `0`–`7` controlling kanji treatment. See [transliterate_japanese.py](script/transliterate_japanese.py) for the conversion logic.

Each configuration has three text files:

| File | Hugging Face split | Source directory |
| --- | --- | --- |
| `train.txt` | `train` | `ReazonSpeech/100m/` |
| `dev.txt` | `validation` | `ReazonSpeech/dev/` |
| `test.txt` | `test` | `ReazonSpeech/test/` 

### Dataset license

The dataset is distributed under the [Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)](https://cdla.dev/sharing-1-0/). This license statement applies to the dataset.

## Repository contents

| Directory | Contents |
| --- | --- |
| `script/` | Corpus sampling, script conversion, tokenizer training, language model training, and evaluation scripts |
| `config/` | Model configurations |
| `tokenizer/` | Tokenizer artifacts |
| `data/` | Supporting data and evaluation resources |
| `experiment/` | Experiment outputs and evaluation results |
| `paper/` | Paper materials |
