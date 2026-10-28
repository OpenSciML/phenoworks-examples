# PhenoWorks Examples

Example BentoML model services and LgoPy analysis blocks for PhenoWorks.

- [Hello World](hello_world/README.md): Return a greeting for an input filename.
- [Iris Classifier](iris/README.md): Train a flower classifier, serve predictions,
  and process a CSV through a dataset-level block.

## Setup

Install Python 3.11 and [uv](https://docs.astral.sh/uv/), then run from this directory:

```sh
uv sync --locked
```

The committed `uv.lock` records the dependency versions for these examples.
The project is a collection of runnable examples rather than an installable library.

## Run Hello World

Build and serve the model:

```sh
sh hello_world/build.sh
sh hello_world/serve.sh
```

In another terminal at the repository root, run the block example:

```sh
uv run --locked python hello_world/hello_world_block.py
```

It checks the greeting against the running service and builds a block ZIP in
`hello_world/dist/`. The model build exports `hello_world/hello_world.zip`.
The `.http` files provide sample requests for compatible HTTP clients.

## Run Iris

First launch Jupyter and run every cell in `iris/iris_model.ipynb` in order:

```sh
uv run --locked jupyter lab iris/iris_model.ipynb
```

The notebook trains and saves the classifier in `iris/.bentoml/`. After training:

```sh
sh iris/build.sh
sh iris/serve.sh
```

In another terminal at the repository root:

```sh
uv run --locked python iris/iris_block.py
```

This processes the sample `iris/iris.csv` and builds a block ZIP in `iris/dist/`.
The model build exports `iris/iris_classifier.zip`.

Both services use port 4000; run one at a time. Build scripts replace an existing
Bento with the same `0.1.0` tag. Hello World uses the default BentoML store;
Iris uses its local `iris/.bentoml/` store.

## Validation

```sh
make check
```

This checks Python syntax, shell syntax, and consistency between the dependency
lock and project metadata. The block commands above are integration checks that
require their respective running services. There is no automated test suite yet.

Generated archives, local model stores, virtual environments, and editor settings
are ignored. Keep credentials out of notebooks and committed configuration.

## License

A license has not yet been selected for this repository.
