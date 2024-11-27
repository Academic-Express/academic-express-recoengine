# academic-express-recoengine

## Installation

### Requirements

1. [Install PyTorch](https://pytorch.org/get-started/locally/)
2. `pip install "sentence_transformers>=2.7.0"`
3. (Optional) [Install xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers>)
4. `pip install "fastapi[standard]" sqlmodel`

### Download Models

```sh
huggingface-cli download Alibaba-NLP/new-impl \
    --local-dir pretrained_models/Alibaba-NLP/new-impl
huggingface-cli download Alibaba-NLP/gte-large-en-v1.5 \
    --local-dir pretrained_models/Alibaba-NLP/gte-large-en-v1.5 --exclude '*.onnx'
```

## Development

```sh
fastapi dev app/main.py
```
