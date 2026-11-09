"""Prediction semantics exposed through MLflow's PythonModel extension."""

import joblib
import mlflow.pyfunc
import pandas as pd


def prediction_frame(model, inputs):
    if model.spec.mode == "forecast":
        return model.predict(inputs)
    result = pd.DataFrame({"prediction": model.predict(inputs)})
    if hasattr(model, "predict_proba"):
        for index, values in enumerate(model.predict_proba(inputs).T):
            result[f"probability_{index}"] = values
    return result


class PredictionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        if self.model.spec.mode == "tabular" and self.model.spec.task == "regression":
            return self.model.predict(model_input)
        return prediction_frame(self.model, model_input)
