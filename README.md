# ragen

<p align="center">
    <a href="https://pypi.org/project/mosec/">
    <img src="https://badge.fury.io/py/mosec.svg" alt="PyPI version" height="20">
    </a>
    <a href="https://github.com/kemingy/ragen/actions/workflows/check.yml">
    <img src="https://github.com/kemingy/ragen/actions/workflows/check.yml/badge.svg?branch=main" alt="Check status" height="20">
  </a>
</p>

A simplest Retrieval Augmented Generation CLI.

## Installation

```bash
# This will install the GPU version PyTorch that might be slow, you can install the CPU
# version to avoid this.
# `pip install torch --index-url https://download.pytorch.org/whl/cpu`
pip install ragen
```

## Usage

```bash
ragen --help
# use the essay in this repo as an example
ragen --data essay_what_work_on.txt --api-key YOUR_OPENAI_API_KEY --top-k 5
```
