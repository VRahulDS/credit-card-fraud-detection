from kedro.pipeline import Node, Pipeline
from .nodes import clean_data, create_features, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=clean_data,
                inputs="fraud_raw",
                outputs="fraud_processed",
                name="clean_data_node",
            ),
            Node(
                func=create_features,
                inputs="fraud_processed",
                outputs="fraud_features",
                name="feature_engineering_node",
            ),
            Node(
                func=split_data,
                inputs="fraud_features",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )
