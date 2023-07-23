import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5001")
mlflow.set_experiment("tutorial")

for i in range(3):
    mlflow.log_param("trial", i)
    mlflow.log_metric("metric", i + 1)
