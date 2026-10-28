#!/bin/sh
set -eu

cd "$(dirname "$0")"

MODEL_TAG="hello_world:0.1.0"

uv run bentoml serve "$MODEL_TAG" --port 4000
