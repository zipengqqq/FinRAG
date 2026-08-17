# 本地 Docker 中间件方案

## 目标

让 FinRAG 在 Windows 上以本机 Python/FastAPI 进程运行，且其依赖的
MinIO、Milvus 和 etcd 通过 Docker Compose 在本机启动。项目复用用户已经
运行的 MySQL 容器，不创建或管理该容器。所有中间件均使用全新的开发数据；
不迁移服务器上的任何数据。

## 范围

本次修改包含：

- 将仓库已有的 MinIO、etcd、Milvus Compose 配置整理为适合 Windows 本地开发
  的启动方式。
- 将 Python 中的 Milvus、MinIO 和 MySQL 连接改为本地环境变量配置。
- 为已有 MySQL 容器提供可重复执行的 `fin_rag` 数据库和 `file` 表初始化方式。
- 提供安全的环境变量模板及 Windows 启动、停止和验证说明。

本次不包含：

- 新建、停止或删除用户已有的 MySQL Docker 容器。
- 从服务器迁移数据。
- 将 FastAPI 或前端放入 Docker。
- 修改 RAG 模型、解析流程或应用接口。

## 架构

```text
Windows 本机 Python / FastAPI
  |-- MySQL（用户已有的 Docker 容器，通过 127.0.0.1:MYSQL_PORT 访问）
  |-- MinIO（Docker Compose，127.0.0.1:9000；控制台 9001）
  `-- Milvus（Docker Compose，127.0.0.1:19530）
        |-- etcd（仅 Docker 网络内部使用）
        `-- MinIO（Docker 网络内部使用）
```

Compose 只管理 `minio`、`etcd` 和 `standalone`（Milvus）。服务使用 Docker
命名卷保存本机开发数据，以规避 Windows 宿主目录挂载的权限和性能问题。通过
健康检查与 `depends_on` 条件保证 Milvus 在依赖就绪后启动。

## 配置与初始化

`.env` 是开发者本机私有文件，保留 API 密钥及实际 MySQL 认证信息，不再由 Git
跟踪。`.env.example` 仅保留变量名称与安全的本地示例：

- MySQL：`DATABASE_URI` 指向用户已有的、映射到 `127.0.0.1` 的 MySQL 容器。
- MinIO：`ENDPOINT=127.0.0.1:9000`，以及本地 MinIO 的访问凭据和 bucket 名称。
- Milvus：使用 `MILVUS_HOST=127.0.0.1`、`MILVUS_PORT=19530`。

应用删除写死的服务器 Milvus 地址，所有服务连接均从环境变量读取。上传路径会
继续在首次使用时创建 MinIO bucket；向量写入时继续在首次使用时创建 Milvus
collection。

提供一个幂等数据库初始化入口。它只执行 `CREATE DATABASE IF NOT EXISTS fin_rag`
及 `file` 表的创建或确认操作，因此可安全地重复执行，且不会修改 MySQL 容器中
的其他数据库、表或记录。

## 启动与失败处理

标准启动顺序如下：

1. 确认已有 MySQL 容器正在运行，且端口映射可从 Windows 访问。
2. 从 `.env.example` 创建本机 `.env` 并填写 MySQL 和 API 密钥。
3. 执行 `docker compose up -d` 启动 MinIO、etcd、Milvus。
4. 执行数据库初始化入口。
5. 用 Windows Python 环境运行 FastAPI。

若 Compose 服务未就绪，健康检查和连接检查应明确显示具体服务；应用不会尝试
回退连接到服务器。数据库初始化失败时应保留既有数据且以非零退出；MinIO bucket
和 Milvus collection 的延迟创建失败将由现有上传/入库流程记录为失败状态。

## 验证

实施完成后应验证：

- `docker compose config` 可以解析配置，三个 Compose 服务均处于健康或运行状态。
- Windows 主机可访问 MinIO 的 API 与控制台、Milvus 的 19530 端口。
- 数据库初始化可以重复执行，并且只存在/确认 `fin_rag.file`。
- Python 环境能成功连接 MySQL、MinIO、Milvus；代码中不再存在服务器 Milvus
  地址。
- README 所列 PowerShell 启动流程可以从全新本地中间件状态完成启动。

## 取舍

不把 MySQL 加入 Compose，因为用户已经有可用容器；这避免端口、卷和已有数据的
冲突。代价是 MySQL 容器的启动仍由用户负责，README 会明确这一前置条件。
