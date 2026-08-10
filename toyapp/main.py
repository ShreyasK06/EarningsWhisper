from fastapi import FastAPI
import os
import socket

app = FastAPI(title="toyapp")


@app.get("/")
def root():
    return {"message": "hello from toyapp", "hostname": socket.gethostname()}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/env")
def env():
    return {"PORT": os.environ.get("PORT", "8000")}
