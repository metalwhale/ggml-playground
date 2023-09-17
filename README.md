# ggml-playground
ML model inference in Zig using ggml

## How to run
1. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it ggml-playground bash
    ```
2. Convert models:
    ```bash
    cd ./models/
    python3 ./convert.py gpt_neox "rinna/japanese-gpt-neox-small"
    ```
3. Inference:
    ```bash
    cd ../ggml-playground/
    zig build test
    ```
