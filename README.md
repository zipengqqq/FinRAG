# FinRAG

FinRAG 是一个面向通用文档的检索增强生成（RAG）服务。系统支持 PDF 文档解析、Markdown 结构化切分、向量与关键词混合检索、重排序、连续记录恢复，以及基于检索上下文的流式问答。

## 核心流程

```text
上传文件或文本
    -> PDF 解析为 Markdown
    -> 按标题、段落和表格切分为子块
    -> 写入 Milvus 向量库和 SQLite FTS5 关键词索引

用户查询
    -> 基于对话历史改写检索问句
    -> Milvus 向量召回 Top 40
    -> SQLite FTS5 / BM25 关键词召回 Top 40
    -> RRF 融合、Reranker 重排序
    -> 按 parent_id 恢复完整连续记录
    -> 大模型基于上下文流式生成答案
```

## 关键技术

- **FastAPI**：提供文件上传、文档管理、检索和问答 API。
- **LangChain 与 LangGraph**：组织查询改写、检索、生成的 RAG 工作流。
- **Milvus**：使用 HNSW/IP 索引保存向量，实现语义相似度检索。
- **BGE-M3**：生成归一化文本向量。
- **SQLite FTS5**：持久化关键词索引，使用 BM25 独立进行关键词召回。
- **RRF（Reciprocal Rank Fusion）**：融合向量检索与关键词检索的排序结果，避免直接比较不同检索器的原始分数。
- **BGE Reranker**：对融合候选进行交叉编码器重排序。
- **Marker**：将 PDF 转换为 Markdown，并支持 OCR 与图片提取配置。
- **MinIO**：保存原始上传文件及解析产物。
- **MySQL + SQLAlchemy**：保存文档元数据和处理状态。
- **Docker Compose**：启动 Milvus、MinIO、etcd 等本地中间件。

## 主要功能

### 文档处理与入库

- 支持上传 PDF 等文件，并在后台完成文件存储、解析、切分和入库。
- 支持直接提交文本，通过 `/insert` 接口写入知识库。
- Markdown 按标题和正文结构切分，保留来源、章节、文档标识等元数据。
- 支持跨页或连续表格识别与合并，避免表格被错误拆散。
- 每个连续记录的子块共享 `parent_id`，并保存 `chunk_index` 与 `chunk_count`。
- 写入 Milvus 成功的子块会同步写入 SQLite 关键词索引；同一稳定键重复写入会覆盖旧索引记录。

### 混合检索

- **向量检索**：Milvus 使用 BGE-M3 向量进行语义召回。
- **关键词检索**：SQLite FTS5 独立执行 BM25 检索，不依赖向量候选集。
- **通用分词**：英文、数字、下划线连续串作为词元；中文使用二元词，并保留单字词元。
- **RRF 融合**：按 `parent_id + chunk_index` 去重并融合两路召回结果。
- **重排序**：融合候选由 Reranker 统一排序，提高精确匹配和语义匹配的最终相关性。
- **元数据过滤**：检索支持按 `source` 和 JSON `metadata` 的精确等值过滤。
- **连续内容恢复**：命中任一子块后，按 `parent_id` 读取并按 `chunk_index` 返回该记录的所有子块。
- **故障降级**：关键词索引不可用或查询失败时，检索自动保留向量检索路径。

### 智能问答

- 结合会话历史改写不完整或带指代的检索问句。
- 基于召回内容生成回答；无有效上下文时使用通用回答路径。
- 通过 Server-Sent Events（SSE）流式返回问答结果。

### 文件管理

- 查询文档列表及处理状态。
- 下载原始文件。
- 预览原始文件或解析后的 Markdown 内容。

## API 概览

| 接口 | 说明 |
| --- | --- |
| `POST /upload_file` | 上传文件并后台解析、切分和入库。 |
| `POST /document` | 查询文档列表及状态。 |
| `POST /file_download` | 下载原始文件。 |
| `POST /file_preview` | 预览原始文件或解析产物。 |
| `POST /insert` | 将文本切分后写入知识库。 |
| `POST /search` | 执行混合检索，返回内容、重排分数和元数据。 |
| `POST /assistant` | 执行带会话历史的 RAG 流式问答。 |

## 本地开发

### 前置条件

- 已安装 Python 及 `requirements.txt` 中的依赖。
- Docker Desktop 正在运行。
- 已准备可访问的 MySQL 服务。MySQL 由用户自行管理，不由 Docker Compose 启动。
- 已准备大模型服务所需的 API Key。

### 配置环境变量

在仓库根目录创建本地环境文件：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

- 配置 `DATABASE_URI`，指向 MySQL 用户、密码、主机、端口和数据库。
- 配置 `DEEPSEEK_API_KEY`；按需配置 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。
- 按本地环境调整 MinIO 配置。
- 可选配置 `MILVUS_HOST` 和 `MILVUS_PORT`。

### 启动中间件

```powershell
docker compose up -d
docker compose ps
```

启动的本地中间件包括 Milvus、MinIO 和 etcd。MinIO 控制台地址为 `http://127.0.0.1:9001`，使用 `.env` 中的 `ACCESS_KEY` 和 `SECRET_KEY` 登录。

### 初始化数据库并启动服务

确认 `.env` 指向目标 MySQL 实例后，初始化数据库：

```powershell
python -m scripts.init_local_db
```

启动 API：

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8288
```

服务启动后可访问 `http://127.0.0.1:8288/docs` 查看 OpenAPI 文档。

### 停止中间件

```powershell
docker compose down
```

上述命令会停止 Compose 容器和网络，但保留命名卷。若需同时删除 Compose 创建的命名卷：

```powershell
docker compose down -v
```

这两个命令都不会管理或删除外部 MySQL 服务及其数据。

## 历史关键词索引重建

升级到包含关键词检索的版本后，已有的 Milvus 数据不会自动出现在 SQLite FTS5 索引中。执行以下命令从 Milvus 全量重建关键词索引：

```powershell
python scripts/rebuild_bm25_index.py
```

可选参数：

```powershell
python scripts/rebuild_bm25_index.py --database-path data/keyword_index.db --page-size 500
```

脚本使用临时 SQLite 文件构建索引，所有分页成功后再原子替换正式索引；重建失败时保留旧索引。

## 项目结构

- `main.py`：FastAPI 应用与 HTTP 接口。
- `service/`：文件处理、文档管理与流式对话服务。
- `marker_parse.py`：PDF 解析为 Markdown。
- `chunker.py`：Markdown 结构化切分与表格连续记录处理。
- `vector_store.py`：Milvus 向量库初始化与文档写入。
- `keyword_index.py`：SQLite FTS5 关键词索引读写。
- `retriever.py`：向量检索、关键词检索、RRF 融合、重排序和连续记录恢复。
- `rag_graph.py`：查询改写、检索、回答生成工作流。
- `scripts/`：数据库初始化与关键词索引重建脚本。
- `tests/`：自动化测试。
