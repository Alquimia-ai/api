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

# FastAPI with Redis Docker Compose Documentation

Integrate a FastAPI application with Redis using Docker and Docker Compose.

## Overview

When deploying services like FastAPI and Redis in Docker containers, it's important to ensure they can communicate. The following steps guide you in setting up FastAPI and Redis using Docker Compose and ensuring they can connect seamlessly.

## Steps

### 1. Docker Compose Configuration

In your `docker-compose.yml`, ensure you have services for both FastAPI and Redis.

```yaml
version: '3'
services:
  redis:
    image: redis
    ports:
      - '6379:6379'
    command: redis-server --bind 0.0.0.0
  fastapi-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - '80:80'
    environment:
      - QDRANT_API_KEY=your_qdrant_api_key
      - OPENAI_API_KEY=your_openai_api_key
      - QDRANT_URL=your_qdrant_url
      - REDIS_URL=redis://redis:6379/0
```

Replace your_qdrant_api_key, your_openai_api_key, and your_qdrant_url with your actual keys and URLs.
