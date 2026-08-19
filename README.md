# FinRAG

## Windows Local Development

### Prerequisites

- Python and the dependencies in `requirements.txt` are installed.
- Docker Desktop is running.
- A MySQL server is running and reachable from this machine. MySQL is external and user-managed: it is not started by `docker compose`.

### Start the local services

In PowerShell at the repository root, create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` before continuing:

- Set `DATABASE_URI` with the MySQL user, password, host, port, and database name for your user-managed MySQL server.
- Set `DEEPSEEK_API_KEY` and, when needed, `DEEPSEEK_BASE_URL` for your API account.
- Adjust the MinIO credentials only when the defaults are not appropriate for your local environment.

Start the local Milvus, MinIO, and etcd middleware:

```powershell
docker compose up -d
docker compose ps
```

The MinIO console is available at `http://127.0.0.1:9001`. Use the `ACCESS_KEY` and `SECRET_KEY` from `.env` to sign in.

Initialize the database only after `.env` points to the intended local MySQL instance:

```powershell
python -m scripts.init_local_db
```

Start the API on loopback port 8288:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8288
```

### Stop local middleware

```powershell
docker compose down
```

This removes the Compose containers and network while retaining the Compose named volumes. To also remove only the named volumes created by this Compose project, run:

```powershell
docker compose down -v
```

Neither command manages or removes the external MySQL server or its data.

## Project Files

- `main.py`: FastAPI application entry point.
- `download_model.py`: downloads the reranking and embedding models.
- `marker_parse.py`: converts PDF documents to Markdown.
- `chunker.py`: splits Markdown content into `List[Document]` chunks.
- `vector_store.py`: stores document chunks in Milvus.
- `rag_graph.py`: defines the RAG retrieval workflow.
- `test.py` and `tests/`: test utilities and automated tests.

## 本项目职责
本项目知识用来学习RAG项目，有些地方会存在小问题，但只要不涉及RAG核心，即可忽略