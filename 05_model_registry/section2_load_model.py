import os

import mlflow
import pandas as pd
from minio import Minio

BUCKET_NAME = "raw-data"
OBJECT_NAME = "iris"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


def download_data():
    #
    # minio client
    #
    url = "0.0.0.0:9000"
    access_key = "minio"
    secret_key = "miniostorage"
    client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)

    #
    # data download
    #
    object_stat = client.stat_object(BUCKET_NAME, OBJECT_NAME)
    data_version_id = object_stat.version_id
    client.fget_object(BUCKET_NAME, OBJECT_NAME, file_path="download_data.csv")
    return data_version_id


def load_data():
    data_version_id = download_data()
    df = pd.read_csv("download_data.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    data_dict = {"data": X, "target": y, "version_id": data_version_id}
    return data_dict


def load_pyfunc_model(run_id, model_name):
    clf = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")
    return clf


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--model-name", type=str, default="my_model")
    args = parser.parse_args()

    #
    # load data
    #
    data_dict = load_data()
    X = data_dict["data"]
    #
    # load model
    #
    pyfunc_clf = load_pyfunc_model(args.run_id, args.model_name)
    pyfunc_pred = pyfunc_clf.predict(X)
    print("pyfunc")
    print(pyfunc_clf)
    print(pyfunc_pred)
