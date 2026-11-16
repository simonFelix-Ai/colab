# 计算单元、缓存与持久化

推荐顺序：本地写代码和提示词 → 上传 GitHub → Colab CPU 下载依赖和模型 → 结束 CPU 会话 → G4 拉取固定代码提交 → 离线恢复环境 → 资源预检 → smoke → 正式推理 → 保存并释放运行时。

## 哪些工作已经提前完成

仓库包含可直接打开的 notebook、下载器、配置检查、固定模型文件清单、推理入口、生成配置和中文排错文档。CPU 准备阶段会下载用于 Linux x86_64 / Python 3.12 的 CUDA 安装包，构建纯 Python 的固定 Diffusers 源码版本，检查其 API，生成精确依赖版本清单，并下载所选任务的权重。

Windows 上的虚拟环境或 Windows wheel 不能直接给 Linux Colab 使用，所以环境缓存必须在 CPU Colab 或等价 Linux 环境生成，而不是把本机 `.venv` 上传。

## 哪些开销仍留在 GPU 阶段

新虚拟机必须恢复 Linux 虚拟环境、读取/反序列化权重、将组件放到 GPU、分配激活与编解码缓冲区。输出保存到 Drive 也有 I/O 成本。缓存安装包只能省网络下载和源码准备，不能把安装与加载变成零耗时。

不要在连接 G4 后长时间编辑提示词或等待回答；先在本地把配置与素材准备好。结束时只关闭浏览器标签未必释放运行时，应主动断开并删除或运行 `runtime.unassign()`。此仓库不做保活，不绕过 Colab 使用限制。

## Drive 中保存什么

默认根目录：`MyDrive/minimax-h3-colab`。

| 路径 | 内容 |
|---|---|
| `environment-py312-<依赖指纹>/wheels` | CUDA 与 Python 安装包 |
| `environment-py312-<依赖指纹>/resolved.txt` | 实际解析出的精确版本 |
| `environment-py312-<依赖指纹>/environment.json` | Python、平台、依赖文件指纹 |
| `models/MiniMax-H3` | 固定 revision 的 Diffusers 权重及 Hugging Face 下载元数据 |
| `outputs/*.mp4` | 带原生音频的输出 |
| `outputs/*.json` | 参数、环境信息、状态、耗时、错误 |

权重目录中的 `.cache/huggingface` 用于增量下载，保留它。首次约需 144GB 下载量，权重不会存放在 GitHub。Drive 基础免费容量通常无法容纳，先检查你自己的可用配额。

## 缓存失效规则

更改 `requirements-colab.txt` 会得到新指纹，需要在 CPU 阶段重新缓存。Python 小版本或平台改变时会停止恢复，防止错误安装。模型文件大小或版本标记不符时会停止 GPU 推理，返回 CPU 下载器修复。推理进程设置 Hugging Face 离线模式；不在 GPU 会话中临时自动下载缺失的百 GB 权重。

官方来源：[Colab FAQ](https://research.google.com/colaboratory/faq.html)。资源与限制动态变化，当前计算单元速率以账号运行时界面为准。
