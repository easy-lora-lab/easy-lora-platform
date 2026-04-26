# EasyTune 微调傻瓜包

EasyTune 是围绕现有微调框架的产品化外壳。第一版只做本地私有化部署，不做 SaaS、计费、支付和复杂权限。

当前 MVP 跑通这条链路：

```text
上传数据集 -> 数据质检 -> 转换 LLaMA-Factory 数据 -> 生成训练配置
-> 创建训练任务 -> 真实 qwen LLaMA-Factory 训练或 mock train
-> 采集日志 -> 保存输出目录 -> 生成模型版本 -> 人工验收记录
```

## 模型方向

第一版只预留两个模型方向：

- `qwen`：当前按 LLaMA-Factory 训练配置生成 YAML，检测到 `llamafactory-cli` 时执行 `llamafactory-cli train {config_path}`，否则自动 mock。
- `rwkv`：当前只生成 RWKV runner 占位配置和命令 `rwkv-finetune train {config_path}`，始终可通过 mock train 跑通业务流程。

后续可以让两个实习生分别接入：

- Qwen 实习生：完善 LLaMA-Factory 的 qwen 模板、显存参数、真实训练前置检查和失败解析。
- RWKV 实习生：替换 `training_service.py` 中的 RWKV 占位执行逻辑，接入真实 RWKV 微调命令和配置生成。

## 目录结构

```text
easytune/
  backend/
    app/
      routers/
      services/
      storage/
  frontend/
  docker-compose.yml
  README.md
```

`storage/` 会自动创建并保存上传文件、转换数据、配置、输出、日志和报告。仓库只保留 `.gitkeep`，运行生成物不会提交。

## 本地后端启动

```bash
cd easytune/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/api/health
```

## 本地前端启动

```bash
cd easytune/frontend
npm install
npm run dev
```

打开：

```text
http://localhost:3000
```

## Docker Compose 启动

```bash
cd easytune
docker compose up --build
```

访问：

```text
前端：http://localhost:3000
后端：http://localhost:8000
```

## 使用流程

1. 进入「数据集管理」，上传 `.json`、`.jsonl` 或 `.csv`。
2. 查看质检报告，包括空文件、行数、空行、行长度、格式识别和质量分。
3. 点击「转换」，生成 `storage/llamafactory_data/dataset_{id}.jsonl` 和 `dataset_info.json`。
4. 进入「创建训练任务」，选择已转换数据集，选择 `qwen` 或 `rwkv`。
5. 提交后查看任务详情页，可见 `command`、`config_path`、`log_path`、`output_dir`。
6. 点击启动。没有 `llamafactory-cli` 时不会崩溃，会自动进入 mock train。
7. 训练完成后自动生成 ModelVersion。
8. 进入「人工验收」，选择模型版本，保存人工评测记录。

## API

```text
POST /api/datasets/upload
GET  /api/datasets
GET  /api/datasets/{id}
POST /api/datasets/{id}/convert
GET  /api/datasets/{id}/dataset-info

POST /api/training-jobs
GET  /api/training-jobs
GET  /api/training-jobs/{id}
POST /api/training-jobs/{id}/start
GET  /api/training-jobs/{id}/logs

GET  /api/model-versions
GET  /api/model-versions/{id}

POST /api/validation-records
GET  /api/validation-records

GET  /api/health
```

## 后续接入点

- `backend/app/services/dataset_service.py`：增强数据清洗、字段映射、采样预览和质检规则。
- `backend/app/services/llamafactory_config_service.py`：扩展 qwen/RWKV 不同训练配置模板。
- `backend/app/services/training_service.py`：替换 mock runner，接入真实训练进程、GPU 检查和中断能力。
- `backend/app/services/model_service.py`：接入导出、模型注册、MLflow 或文件校验。
- `backend/app/services/validation_service.py`：后续接 Open WebUI、vLLM、Ollama 的真实推理结果。

## LLaMA-Factory 参考目录

仓库根目录的 `LlamaFactory/` 只作为本地代码参考，不上传 GitHub；根 `.gitignore` 已忽略该目录。
EasyTune 不 import 或读取这个本地目录；qwen 真实训练只检测系统 PATH 里的 `llamafactory-cli`，没有 CLI 时自动走 mock train。
