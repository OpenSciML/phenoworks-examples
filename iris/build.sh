#!/bin/sh
set -eu

cd "$(dirname "$0")"

MODEL_TAG="iris_classifier:0.1.0"
export BENTOML_HOME="$PWD/.bentoml"

if uv run bentoml get "$MODEL_TAG" > /dev/null 2>&1; then
    uv run bentoml delete "$MODEL_TAG" --yes
fi

uv run bentoml build --version "${MODEL_TAG##*:}"
uv run bentoml export "$MODEL_TAG" "$PWD/iris_classifier.zip"
