#!/bin/sh
set -eu

cd "$(dirname "$0")"

MODEL_TAG="iris_classifier:0.1.0"
export BENTOML_HOME="$PWD/.bentoml"

uv run bentoml serve "$MODEL_TAG" --port 4000
