"""MLflow Models From Code entrypoint."""

from mlflow.models import set_model
from agent_mlops.inference import PredictionModel

set_model(PredictionModel())
