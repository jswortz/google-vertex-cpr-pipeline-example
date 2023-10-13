
import pandas as pd
import pickle
import torch
from typing import Dict

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils


class CustomPyTorchPredictor(Predictor):
    
    def __init__(self):
        self._class_names = ["setosa", "versicolor", "virginica"]
    
    def load(self, artifacts_uri: str):
        """Loads the model artifacts."""
        prediction_utils.download_model_artifacts(artifacts_uri)

        self._model = torch.load("model.pt")

    def preprocess(self, prediction_input: Dict) -> torch.Tensor:
        instances = prediction_input["instances"]
        data = pd.DataFrame(instances).values
        return torch.Tensor(data)

    @torch.inference_mode()
    def predict(self, instances: torch.Tensor) -> torch.Tensor:
        """Performs prediction."""
        outputs = self._model(instances)
        _ , predicted = torch.max(outputs, 1)
        return predicted

    def postprocess(self, prediction_results: torch.Tensor) -> Dict:
        return {"predictions": [self._class_names[class_num] for class_num in prediction_results]}
