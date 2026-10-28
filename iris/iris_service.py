"""Serve the Iris classifier trained and saved by iris_model.ipynb."""

from typing import Annotated

import bentoml
import numpy as np
import pandas as pd
from bentoml.exceptions import BadInput
from bentoml.validators import DataframeSchema
from pydantic import BaseModel, Field

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Match the training environment so the saved scikit-learn pipeline is compatible.
inference_image = bentoml.images.Image(python_version="3.11", lock_python_packages=False) \
    .python_packages(
        "scikit-learn==1.9.1",
        "pandas==3.1.0",
        "numpy==2.4.6",
        "pydantic==2.13.5",
    )


class FlowerMeasurements(BaseModel):
    """Four positive flower measurements in centimeters used for prediction."""

    sepal_length: float = Field(gt=0, allow_inf_nan=False, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, allow_inf_nan=False, description="Sepal width in cm")
    petal_length: float = Field(gt=0, allow_inf_nan=False, description="Petal length in cm")
    petal_width: float = Field(gt=0, allow_inf_nan=False, description="Petal width in cm")


@bentoml.service(name="iris_classifier", image=inference_image)
class IrisClassifierService:
    """Predict Iris species using the packaged preprocessing and classifier."""

    # BentoML bundles this saved model into the ZIP and makes it available on import.
    model = bentoml.models.BentoModel("iris_classifier:latest")

    def __init__(self) -> None:
        """Load the packaged model and its training feature and species names."""
        self.classifier = bentoml.sklearn.load_model(self.model)
        self.feature_names: list[str] = list(self.model.info.metadata["feature_names"])
        self.target_names: list[str] = list(self.model.info.metadata["target_names"])

    @bentoml.api
    def predict(self, measurements: FlowerMeasurements) -> dict[str, int | str]:
        """Classify one flower from four measurements in centimeters.

        Args:
            measurements: Validated sepal and petal lengths and widths.

        Returns:
            Integer class ID and the corresponding Iris species name.
        """
        return self._predict_measurements(pd.DataFrame([measurements.model_dump()]))[0]

    @bentoml.api
    def predict_batch(
        self,
        measurements: Annotated[
            pd.DataFrame,
            DataframeSchema(orient="records", columns=FEATURE_COLUMNS),
        ],
    ) -> list[dict[str, int | str]]:
        """Classify a batch of flowers with one model prediction call.

        Args:
            measurements: DataFrame with the four measurement columns in
                centimeters. JSON requests use a list of row objects.

        Returns:
            One class ID and species per flower, preserving input order.
            An empty batch returns an empty list.

        Raises:
            BadInput: If any measurement is missing, nonnumeric, nonfinite,
                or not positive.
        """
        return self._predict_measurements(measurements)

    def _predict_measurements(
        self, measurements: pd.DataFrame
    ) -> list[dict[str, int | str]]:
        """Run the shared inference logic without calling another API endpoint.

        Args:
            measurements: Flower measurements in input order.

        Returns:
            Ordered predictions, or an empty list for an empty batch.

        Raises:
            BadInput: If measurements are missing or not positive finite numbers.
        """
        if measurements.empty:
            return []
        try:
            frame = measurements[FEATURE_COLUMNS].astype(float)
        except (KeyError, TypeError, ValueError) as error:
            raise BadInput("Provide all four numeric flower measurements.") from error
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise BadInput("Flower measurements must be positive finite numbers.")
        # Match the feature names and order recorded during training.
        frame.columns = self.feature_names
        return [
            {"class_id": int(class_id), "species": self.target_names[int(class_id)]}
            for class_id in self.classifier.predict(frame)
        ]
