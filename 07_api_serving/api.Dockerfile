FROM amd64/python:3.9-slim

WORKDIR /usr/app/

RUN pip install -U pip &&\
    pip install mlflow==2.3.2

RUN pip install "fastapi[all]"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY downloads/ /usr/app/downloads/

COPY app.py app.py
ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0" ]
