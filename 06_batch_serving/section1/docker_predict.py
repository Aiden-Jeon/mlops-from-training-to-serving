import os

import mlflow
import pandas as pd
from minio import Minio

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-artifact-store:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


def predict(run_id, model_name):
    #
    # load model
    #
    clf = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")

    #
    # minio client
    #
    url = "mlflow-artifact-store:9000"
    access_key = "minio"
    secret_key = "miniostorage"
    client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)

    #
    # get data list to predict
    #
    if "predicted" not in client.list_buckets():
        # 최초 실행시 predicted bucket 생성
        client.make_bucket("predicted")
    not_predicted_list = [objects.object_name for objects in client.list_objects(bucket_name="not-predicted")]
    predicted_list = [objects.object_name for objects in client.list_objects(bucket_name="predicted")]
    to_predict_list = []
    for not_predicted in not_predicted_list:
        if not_predicted not in predicted_list:
            to_predict_list += [not_predicted]

    #
    # predict
    #
    for filename in to_predict_list:
        print("data to predict:", filename)
        # download and read data
        client.fget_object(bucket_name="not-predicted", object_name=filename, file_path=filename)
        data = pd.read_csv(filename)

        # predict
        pred = clf.predict(data)

        # save to minio prediction bucket
        pred_filename = f"pred_{filename}"
        pred.to_csv(pred_filename, index=None)
        client.fput_object(bucket_name="predicted", object_name=filename, file_path=pred_filename)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--model-name", type=str, default="my_model")
    args = parser.parse_args()

    #
    # predict
    #
    predict(args.run_id, args.model_name)
