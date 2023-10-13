

from fastapi import FastAPI, Request

import json
import numpy as np
import os
import logging
import torch


from google.cloud import storage
from predictor import CustomPyTorchPredictor

app = FastAPI()

predictor_instance = CustomPyTorchPredictor()
predictor_instance.load(artifacts_uri = os.environ['AIP_STORAGE_URI'])

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    instances = torch.FloatTensor(instances)
    outputs = predictor_instance.predict(instances)

    return {"predictions": predictor_instance.postprocess(outputs)}
