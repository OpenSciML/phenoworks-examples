"""Predict Iris species for rows in a dataset's iris.csv file."""

from pathlib import Path

import pandas as pd
from lgopy.core import Block, BlockContext, ModelReference
from lgopy.models import HttpModelResolver


class PredictIris(Block):
    """Find iris.csv in a dataset and predict the species of each flower.

    Args:
        model: Deployed Iris classifier reference resolved by the execution host.
        batch_size: Maximum number of CSV rows sent in each request.
    """

    name = "predict_iris"
    display_name = "Predict Iris Species"
    description = "Find iris.csv in a dataset and predict the species of each data batch."
    version = "0.1.0"
    authors = ["PhenoWorks Team"]
    categories = ["Examples", "Model Inference"]
    tags = ["phenoworks", "example", "iris", "classification", "csv", "model-reference"]
    extras = {"supported_inputs": [], "transform_scope": "dataset"}

    def __init__(self, model: ModelReference, batch_size: int = 100) -> None:
        """Select the Iris classifier.

        Args:
            model: Model reference resolved by the execution host.
            batch_size: Positive maximum number of rows per prediction request.

        Raises:
            ValueError: If batch_size is not positive.
        """
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model = model
        self.batch_size = batch_size

    def call(self, context: BlockContext) -> pd.DataFrame:
        """Predict each row in the first iris.csv found in the dataset.

        Args:
            context: Dataset information and dataset_items, each with files
                grouped by data product (for example, "csv"). file_uri is the
                local readable path supplied by PhenoWorks for this call.
                CSV columns must be sepal_length, sepal_width, petal_length,
                and petal_width, with positive finite measurements in centimeters.

        Returns:
            Original CSV rows with predicted_class_id and predicted_species
            columns. Returns an empty DataFrame if iris.csv is absent.

        Raises:
            KeyError: If required measurement columns are missing.
            ValueError: If a measurement is not numeric or the response count
                differs from the input batch size.
            OSError: If the matching CSV cannot be read.
            RuntimeError: If the service rejects a measurement or inference fails.
            TypeError: If the model response is not a list of JSON objects.
        """
        iris_file = next(
            (
                file
                for item in context.input["dataset_items"]
                for files in item["files"].values()
                for file in files
                if Path(file.get("relative_path") or file["file_uri"]).name == "iris.csv"
            ),
            None,
        )
        if iris_file is None:
            raise RuntimeError("No iris.csv file found in the dataset.")

        table = pd.read_csv(iris_file["file_uri"])
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        class_ids: list[int] = []
        species: list[str] = []
        measurements = table[columns].astype(float)
        for start in range(0, len(measurements), self.batch_size):
            # DataframeSchema(orient="records") accepts JSON row objects.
            batch = measurements.iloc[start:start + self.batch_size].to_dict(orient="records")
            response = self.models.invoke(
                self.model,
                "predict_batch",
                json={"measurements": batch},
            )
            response.raise_for_status()
            predictions = response.json()
            if not isinstance(predictions, list) or any(
                not isinstance(prediction, dict) for prediction in predictions
            ):
                raise TypeError("The Iris model response must be a list of JSON objects.")
            if len(predictions) != len(batch):
                raise ValueError("The Iris model must return one prediction per input row.")
            class_ids.extend(prediction["class_id"] for prediction in predictions)
            species.extend(prediction["species"] for prediction in predictions)

        table["predicted_class_id"] = class_ids
        table["predicted_species"] = species
        return table


def generate_zip() -> Path:
    """Build the block with its README and citation into dist.zip.

    Returns:
        Path to the generated block ZIP archive.
    """
    root = Path(__file__).parent
    result = PredictIris.build(
        output_dir=root / "dist",
        format="zip",
        readme_file=root / "README.md",
        citation_file=root / "CITATION.cff",
    )
    return Path(result.archive_path)


def main() -> None:
    """Test the sample CSV against the local service and build the block ZIP.

    Start the Iris service on port 4000 before running this file.
    """
    root = Path(__file__).parent
    model_tag = "iris_classifier:latest"
    resolver = HttpModelResolver({model_tag: "http://localhost:4000"})
    try:
        block = PredictIris(model_tag, batch_size=2)
        block.models = resolver
        # A file without a plot is one dataset item, as in the PhenoWorks tutorial.
        dataset_item = {
            "files": {
                "csv": [
                    {
                        "id": 1,
                        "data_product": "csv",
                        "metadata": {},
                        "relative_path": "iris.csv",
                        "file_uri": str(root / "iris.csv"),
                    }
                ]
            },
        }
        context = BlockContext(
            input={
                "dataset_items": [dataset_item],
            }
        )
        predictions = block.call(context)
        print(predictions.head().to_string(index=False))
        print(f"Predicted {len(predictions)} flowers.")
    finally:
        resolver.close()

    print(f"Generated ZIP: {generate_zip()}")


if __name__ == "__main__":
    main()
