from kedro.pipeline import Node, Pipeline
from .nodes import prepare_features, train_model, evaluate_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                func=prepare_features,
                inputs=["X_train", "X_test"],
                outputs=["X_train_encoded", "X_test_encoded"],
                name="prepare_features_node",
            ),
            Node(
                func=train_model,
                inputs=["X_train_encoded", "y_train"],
                outputs="model",
                name="train_model_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["model", "X_test_encoded", "y_test"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )