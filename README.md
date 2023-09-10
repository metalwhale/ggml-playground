# ggml-playground
ML model inference in Zig using ggml

## How to run
1. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it ggml-playground bash
    ```
2. Convert [Rinna's `japanese-gpt-neox-small`](https://huggingface.co/rinna/japanese-gpt-neox-small) model:
    ```bash
    cd ./models/
    mkdir -p ./gpt_neox/rinna-japanese-gpt-neox
    python3 ./convert.py "rinna/japanese-gpt-neox-small" ./gpt_neox/rinna-japanese-gpt-neox/
    ```
3. Inference:
    ```bash
    cd ../ggml-playground/
    zig build test
    ```
