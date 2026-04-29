# EasyTune 微调傻瓜包

EasyTune 是围绕现有微调框架的产品化控制台。推荐部署方式是前后端分离：前端可以部署到阿里云等公网服务器，后端部署在本机 GPU 机器上，负责数据、训练、推理和模型文件访问。

当前 MVP 跑通这条链路：

```text
Dashboard -> Dataset -> Create Fine-tune -> Training Monitor
-> Evaluation -> Chat Playground -> Model Registry -> Deploy / Inference
```

## 模型方向

第一版只预留两个模型方向：

- `qwen`：按 LLaMA-Factory 训练配置生成 YAML。优先检测系统 `llamafactory-cli`，没有时使用内置 `backend/app/vendor/llamafactory`。
- `rwkv`：按 RWKV-PEFT SFT 格式生成训练数据和配置。优先检测系统 `rwkv-finetune`，没有时使用内置 `backend/app/vendor/rwkv_peft/train.py`。
- `rwkv_lightning`：内置 `backend/app/vendor/rwkv_lightning`，用于启动本机 RWKV 推理服务。

当前已经有基础前置检查、runner 检测、GPU 提示和失败摘要。后续可以继续增强：

- Qwen 实习生：继续扩展不同 qwen 尺寸的模板、显存估算和失败解析规则。
- RWKV 实习生：根据实际 RWKV runner 参数继续细化配置模板。

## 目录结构

```text
easytune/
  backend/
    app/
      routers/
      services/
      storage/
      vendor/
        llamafactory/
        rwkv_lightning/
        rwkv_peft/
  frontend/
  docker-compose.yml
  README.md
```

`storage/` 会自动创建并保存上传文件、转换数据、配置、输出、日志、报告和模型 registry。仓库只保留 `.gitkeep`，运行生成物不会提交。

## 本地后端启动

```bash
cd easytune/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

真实训练/推理机器需要额外安装可选依赖：

```bash
pip install -r requirements-ml.txt
```

健康检查：

```bash
curl http://localhost:8000/api/health
```

可选推理验收环境变量：

```bash
# RWKV Lightning, from built-in vendor
cd easytune/backend/app/vendor/rwkv_lightning
python app.py --model-path /path/to/rwkv-model.pth --port 8001 --password optional-token

export EASYTUNE_INFERENCE_PROVIDER=rwkv_lightning
export EASYTUNE_INFERENCE_BASE_URL=http://127.0.0.1:8001
export EASYTUNE_INFERENCE_API_KEY=optional-token

# Qwen local Transformers
export EASYTUNE_INFERENCE_PROVIDER=qwen_transformers
export EASYTUNE_QWEN_BASE_MODEL=/path/to/qwen-base

# OpenAI-compatible/vLLM/Open WebUI
export EASYTUNE_INFERENCE_PROVIDER=openai_compatible
export EASYTUNE_INFERENCE_BASE_URL=http://localhost:8001/openai/v1
export EASYTUNE_INFERENCE_MODEL=easytune-served-model
export EASYTUNE_INFERENCE_API_KEY=optional-token

# 或 Ollama
export EASYTUNE_INFERENCE_PROVIDER=ollama
export EASYTUNE_INFERENCE_BASE_URL=http://localhost:11434
export EASYTUNE_INFERENCE_MODEL=qwen2.5:7b
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

前端部署到阿里云时：

- 构建时可设置 `NEXT_PUBLIC_API_BASE_URL=http://你的本机后端地址:8000`。
- 也可以进入前端页面「Deploy / Inference」，把后端 API 地址保存到当前浏览器。
- 后端默认开放 CORS，实际公网部署建议在反向代理层限制来源和访问令牌。

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

## 页面设计

| 页面 | 内容 |
| --- | --- |
| Dashboard | GPU 状态、最近训练任务、失败任务、模型数量、数据集数量 |
| Dataset | 上传 JSONL/CSV、格式检查、样本预览、train/valid 切分、错误样本提示 |
| Create Fine-tune | base model、训练方式、LoRA/QLoRA、模板选择、epoch/lr/batch/output |
| Training Monitor | 任务进度、日志、命令、配置、输出目录，后续接 loss 曲线和停止/恢复 |
| Evaluation | benchmark、自定义测试集、base vs finetuned 对比入口 |
| Chat Playground | 调用本机推理后端，做问答对比 |
| Model Registry | checkpoint、adapter、export、版本说明 |
| Deploy / Inference | 前端 API 地址、后端健康、vLLM/Ollama/Transformers/RWKV Lightning 配置 |

## 使用流程

1. 进入「数据集管理」，上传 `.json`、`.jsonl` 或 `.csv`。
2. 查看质检报告，包括空文件、行数、非空行数、空行、行长度、格式识别和质量分。
3. 点击「转换」，生成 `storage/llamafactory_data/dataset_{id}.jsonl` 和 `dataset_info.json`。
4. 进入「创建训练任务」，选择已转换数据集，选择 `qwen` 或 `rwkv`。
5. 提交后查看任务详情页，可见 `command`、`config_path`、`log_path`、`output_dir`。
6. 点击启动。没有对应 runner 时不会崩溃，会自动进入 mock train；有 runner 时会先做配置、数据、路径、GPU 提示等前置检查。
7. 训练完成后自动生成 ModelVersion、export manifest 和 `storage/registry/model_versions.json`。
8. 进入「人工验收」，选择模型版本；可手填实际答案，也可在配置推理服务后生成实际答案并保存人工评测记录。

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
POST /api/validation-records/generate
GET  /api/validation-records

GET  /api/health
```

## 后续接入点

- `backend/app/services/dataset_service.py`：继续增强数据清洗、字段映射、采样预览和质检规则。
- `backend/app/services/llamafactory_config_service.py`：继续扩展 qwen/RWKV 不同训练配置模板。
- `backend/app/services/training_service.py`：继续增强中断、重试和更细的失败解析。
- `backend/app/services/model_service.py`：后续可接 MLflow 或远端模型仓库。
- `backend/app/services/inference_service.py`：已经支持 OpenAI-compatible/vLLM/Open WebUI/Ollama，后续可加更多推理后端。

## LLaMA-Factory 参考目录

仓库根目录的 `LlamaFactory/` 只作为本地代码参考，不上传 GitHub；根 `.gitignore` 已忽略该目录。
EasyTune 不 import 或读取这个本地目录；qwen 真实训练只检测系统 PATH 里的 `llamafactory-cli`，没有 CLI 时自动走 mock train。
