# spec-decoding
## Setup
Clone with submodules:
```bash
git clone --recursive git@github.com:michaelpginn/spec-decoding.git
```

Please install [uv](https://docs.astral.sh/uv/getting-started/installation/) to run.

## Research Questions

1. Do **low-resource languages** face worse speedup factors than high-resource languages?
2. For LR languages, it more effective to use draft models that are created via **knowledge distillation** or trained for **language modeling** on monolingual corpora?
    1. For KD, is it better to use a **quantized model** or **smaller model** with a similar architecture?
    2. For LM, is it better to use a **neural model** or an **n-gram model**?
3. How can we practically implement **draft model routing** for a multilingual language model?

## Links
- [📝 Notes doc](https://docs.google.com/document/d/1GcsLQniqIWbxFAj_zbTSZS0302S73-ZZPJ2WA_w1w9g/edit?usp=sharing)
- [📆 Project timeline](https://www.notion.so/Multilingual-Speculative-Decoding-2bc9f22610ac80a98c0bf2eedb6e3457?source=copy_link)
