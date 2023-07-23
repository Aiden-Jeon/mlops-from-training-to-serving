FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install scikit-learn==1.2.2 pandas==2.0.2

COPY train.py train.py

ENTRYPOINT "/bin/bash"
