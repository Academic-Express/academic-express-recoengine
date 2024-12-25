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

### Modify Configuration

After downloading the models, modify the `auto_map` paths in the `pretrained_models/Alibaba-NLP/gte-large-en-v1.5/config.json` file as follows:

```json
"auto_map": {
    "AutoConfig": "./pretrained_models/Alibaba-NLP/configuration.NewConfig",
    "AutoModel": "./pretrained_models/Alibaba-NLP/modeling.NewModel",
    "AutoModelForMaskedLM": "./pretrained_models/Alibaba-NLP/modeling.NewForMaskedLM",
    "AutoModelForMultipleChoice": "./pretrained_models/Alibaba-NLP/modeling.NewForMultipleChoice",
    "AutoModelForQuestionAnswering": "./pretrained_models/Alibaba-NLP/modeling.NewForQuestionAnswering",
    "AutoModelForSequenceClassification": "./pretrained_models/Alibaba-NLP/modeling.NewForSequenceClassification",
    "AutoModelForTokenClassification": "./pretrained_models/Alibaba-NLP/modeling.NewForTokenClassification"
}
```

To enable the `xformers` library, set `torch_dtype` to `float16`, and `use_memory_efficient_attention` to `True` in the `config.json` file.

### Prepare Access Token

Create a `data/access_token` file containing the access token for the API. The access token should be identical to the one used in the backend configuration.

```sh
mkdir -p data
echo "your_access_token" > data/access_token
```

## Development

```sh
fastapi dev app/main.py
```

The API will be available at `http://localhost:8000`. The Swagger UI will be available at `http://localhost:8000/docs`.

## Deployment

```sh
fastapi run app/main.py --host 0.0.0.0 --port 8000
```
