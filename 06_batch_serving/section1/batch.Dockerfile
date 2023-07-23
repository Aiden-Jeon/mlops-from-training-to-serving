FROM amd64/python:3.9-slim

WORKDIR /usr/app/

RUN pip install -U pip &&\
    pip install mlflow==2.3.2 boto3==1.26.151 minio==7.1.15

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY predict.py predict.py
ENTRYPOINT [ "python", "predict.py", "--run-id" ]
