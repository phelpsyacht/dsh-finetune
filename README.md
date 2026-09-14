# dsh-finetune

面向 DeepSeek Harness的 LLM 微调插件。

dsh-finetune将数据集校验、数据清洗、LoRA SFT 微调、训练监控和适配器验证串成一条可直接复用的工具链。它既可以在本机 CPU/GPU 上运行，也可以通过 SSH 调用远端 GPU 机器，或者对接 OpenAI-compatible 的远端微调 API。

## Features

- **数据集校验**
  - 支持 `chat`、`prompt-completion`、`alpaca`、`sharegpt` 四种格式。
  - 检查 JSONL 合法性、格式一致性、assistant 结尾、字段完整性。
  - 返回行级错误、格式、approximate token 数和预览。

- **数据清洗**
  - 去重、类似去重、长度过滤、重复字符过滤。
  - 清理控制字符、HTML、URL，支持全角转半角。
  - 支持 PII 脱敏：身份证、手机号、邮箱。
  - 支持内联 `/quarantine` 边界样本审查报告。
  - 支持按 source 覆盖阈值。
  - 可选 PPL 过滤、语言检查、语义去重、主题下采样。

- **本地 LoRA SFT 微调**
  - 使用 `torch + transformers + peft`。
  - 不依赖 `trl` / `datasets`，尽量减少框架依赖。
  - 支持 assistant 回复 mask。
  - 训练完成后保存 adapter、tokenizer 和 training state。

- **SSH 远端训练**
  - 自动上传数据集、配置和 trainer 到远端机器。
  - 在远端 GPU 机器上启动训练并轮询日志。
  - 适合本机没有 GPU、但有远端训练机的场景。

- **远端微调任务管理**
  - 适配 OpenAI-compatible `/files` 和 `/fine_tuning/jobs` 接口。
  - 提供任务创建、状态查询、列表、取消和后台监视。

- **安全**
  - API key 只从环境变量读取，不进入 tool 参数和模型可见 transcript。
  - side-effectful 工具声明为非并发安全，方便接入 permission policy。

## Supported Tools

| Tool | Purpose |
|---|---|
| `finetune_dataset_validate` | 校验 JSONL 微调数据集 |
| `finetune_dataset_clean` | 清洗、去重、过滤和脱敏数据集 |
| `finetune_train` | 本地 / SSH LoRA SFT 训练 |
| `finetune_job_create` | 创建远端微调任务 |
| `finetune_job_status` | 查询远端微调任务状态 |
| `finetune_job_list` | 列出远端微调任务 |
| `finetune_job_cancel` | 取消远端微调任务 |
| `finetune_job_watch` | 后台监视远端微调任务 |

## Requirements

- Node.js >= 22
- Python >= 3.10
- Python packages: `torch`, `transformers`, `peft`
- 本地训练时，模型需在 HuggingFace 缓存或本地路径中
- 远端 API 模式需要服务商支持 OpenAI-compatible fine-tuning endpoints
- DSH 宿主需提供 `@deepseek-ai/dsh-tools`（`>= 0.1.5-rc.2`）、`@deepseek-ai/cordis`（`^4.0.2`）、`@deepseek-ai/schemastery`（`^3.18.2`）——三者以 `peerDependencies` 声明，随 DSH 本体升级同步
- 后台监视依赖宿主组合的 jobs 注册表（`@deepseek-ai/dsh-jobs-local`，由 `@deepseek-ai/dsh-base` 组合），但它**不是必需依赖**：插件用 `ctx.get("jobs")` 惰性读取而非声明 `inject`，所以没有注册表时 8 个工具仍全部注册——只是 `finetune_job_watch` 在调用时报错并提示改用 `finetune_job_status`，`finetune_train` 退化为 detached 进程运行

## Quick Start

### 1. 校验数据集

```json
{
  "path": "/data/train.jsonl"
}
```

工具：

```text
finetune_dataset_validate
```

### 2. 清洗数据集

```json
{
  "path": "/data/train.jsonl",
  "out": "/data/train.clean.jsonl"
}
```

工具：

```text
finetune_dataset_clean
```

### 3. 本地 LoRA 微调

```json
{
  "dataset_path": "/data/train.clean.jsonl",
  "name": "my-first-lora",
  "model": "Qwen/Qwen3-0.6B",
  "epochs": 2,
  "device": "cpu"
}
```

工具：

```text
finetune_train
```

### 4. SSH 远端训练

在插件配置中设置：

```yaml
transport: ssh
remote:
  host: cp.example.com
  user: ubuntu
  port: 22
  keyPath: ~/.ssh/id_rsa
  dir: /data/dsh-finetune
  python: python3
```

之后调用 `finetune_train` 即可。

## Project Structure

```text
dsh-finetune/
├── index.js                  # 插件注册入口
├── cordis.patch.yml          # Bundle patch
├── lib/
│   ├── clean.js              # 数据集清洗引擎
│   ├── dataset.js            # 数据集格式校验
│   ├── provider.js           # OpenAI-compatible fine-tuning adapter
│   ├── python.js             # Python / torch 环境探测
│   ├── remote.js             # SSH 远端训练执行器
│   ├── tools/
│   │   ├── dataset-validate.js
│   │   ├── dataset-clean.js
│   │   ├── train.js
│   │   ├── job-create.js
│   │   ├── job-status.js
│   │   ├── job-list.js
│   │   ├── job-cancel.js
│   │   └── job-watch.js
│   └── trainer/
│       ├── train.py          # 独立 LoRA SFT trainer
│       └── clean.py          # 可选 Python 清洗层
└── tests/
```

## Note

- 训练方法仅支持 **LoRA**，暂不支持 QLoRA / 全参微调,有扩展能力。
- 本机 CPU 可以训练小模型，但速度较慢；推荐使用 GPU 或 SSH 远端机器。
- 本地 trainer 使用 `HF_HUB_OFFLINE=1`，模型必须已缓存到本地或指定本地模型路径。
- 远端 API 模式依赖服务商是否真的提供 OpenAI-compatible fine-tuning endpoints。
- loss 下降只能说明训练链路在拟合数据，不能替代 held-out 评估和人工评测。

## License

MIT
