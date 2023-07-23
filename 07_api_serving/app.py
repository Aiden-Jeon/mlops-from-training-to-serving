import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


#
# load model
#
model = mlflow.pyfunc.load_model("./downloads/my_model")


#
# Data in and out schema
#
class PredictIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictOut(BaseModel):
    iris_class: str


#
# Write fastapi app
#

app = FastAPI()


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    df.columns = df.columns.str.replace("_", " ")
    df = df.add_suffix(" (cm)")
    pred = model.predict(df).item()
    return PredictOut(iris_class=pred)
