# Iris Classifier

The Iris model predicts setosa, versicolor, or virginica from four flower
measurements in centimeters: sepal length, sepal width, petal length, and petal width.
It returns a numeric class ID and species name.

The dataset-level `PredictIris` block finds the first file named `iris.csv` across
the dataset's items. It expects the columns `sepal_length`, `sepal_width`,
`petal_length`, and `petal_width`, with positive finite numeric values. The sample
`iris.csv` contains all 150 flowers from scikit-learn's Iris dataset, with
50 flowers from each species and measurements in centimeters.

The block sends rows to the model's `predict_batch` endpoint in batches of up to
100 rows (configurable with `batch_size`) and returns the original table with
`predicted_class_id` and `predicted_species` columns. If no `iris.csv` is present,
it returns an empty table.

The batch endpoint accepts a pandas DataFrame declared with
`Annotated[pd.DataFrame, DataframeSchema(orient="records", columns=...)]`.
Over HTTP, pass the rows as `{"measurements": [{...}, {...}]}`.

The local `BlockContext` example follows the
[PhenoWorks custom-block tutorial](https://phenoworks.org/docs/tutorials/build-custom-lgopy-blocks):
dataset scope provides `dataset_id`, `dataset`, `modalities`, `data_products`, and
`dataset_items`. Each item groups records in `files` by data product (`csv`).
For this file without a plot, `plot_id` and `plot` are `None`. The CSV has
modality `none` because no sensor is specified. The block matches the filename
using `relative_path` and reads the local path from `file_uri`.
