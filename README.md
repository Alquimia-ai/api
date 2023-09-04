# FastAPI Server

This is a FastAPI server template. It's a simple guide to get you started quickly.

## Installation

To run this FastAPI server, you'll need to install the required Python packages. You can do this using pip:

```shell
pip install fastapi uvicorn motor dnspython langchain openai PyPDF2 faiss-cpu tiktoken qdrant-client python-multipart redis
```

## Environment variables

Set the following environment variables:

```shell
COMPOSE_PROJECT_NAME
```

in .env if you wish to change the docker container name

### Running server

```shell
python main.py
```
