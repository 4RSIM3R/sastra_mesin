import json
import os
from ray import serve
from starlette.requests import Request
from typing import Dict
import pickle


@serve.deployment(health_check_timeout_s=100, health_check_period_s=100)
class BoostingModel:
    def __init__(self, model_path: str, label_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(label_path) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}


MODEL_PATH = os.path.join(
    "./models/", "iris_model_gradient_boosting_classifier.pkl"
)

LABEL_PATH = os.path.join("./models/", "iris_labels.json")

boosting_model = BoostingModel.bind(
    MODEL_PATH, LABEL_PATH)
serve.run(boosting_model)
