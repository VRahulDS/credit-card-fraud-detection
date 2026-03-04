"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from credit_card_fraud_detection.pipelines.data_processing.pipeline import create_pipeline as dp_create
from credit_card_fraud_detection.pipelines.data_science.pipeline import create_pipeline as ds_create


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines(raise_errors=True)
    # pipelines["__default__"] = sum(pipelines.values())

    dp_pipeline = dp_create()
    ds_pipeline = ds_create()

    pipelines = {
        "dp": dp_pipeline,
        "ds": ds_pipeline,
        "__default__": dp_pipeline + ds_pipeline,
    }
    return pipelines
