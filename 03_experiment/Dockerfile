FROM amd64/python:3.9-slim

RUN pip install -U pip &&\
    pip install mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0"]
