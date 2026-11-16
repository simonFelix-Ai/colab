# MiniMax H3 · Colab G4 96GB

提前准备代码、Linux 安装包和模型权重，在 Colab G4 上拉取 GitHub 后执行本地视频与原生音频推理。

**先运行 CPU 准备笔记本，再运行 GPU 推理笔记本。** 默认文生视频；也提供首尾帧图生视频配置。

本项目固定 MiniMax 官方权重提交与包含 H3 的 Diffusers 提交。当前环境没有 CUDA GPU，**尚未在 Colab G4 完成真实出片验证**；不要把源码/API 核对或单元检查当作 GPU 实测。第一段短片用于验证你的实际运行时。

## 开始使用

1. 点击 [在 Colab 打开 CPU 准备笔记本](https://colab.research.google.com/github/simonFelix-Ai/colab/blob/main/notebooks/01_prepare_cpu.ipynb)，选择 **CPU、Python 3.12**，按顺序运行。仓库地址已填写，模型与安装包保存到 Drive。
2. CPU 准备成功后，点击 [在 Colab 打开 G4 推理笔记本](https://colab.research.google.com/github/simonFelix-Ai/colab/blob/main/notebooks/02_infer_g4.ipynb)，选择 **G4 96GB、Python 3.12**。保持相同 `PERSIST_ROOT`；建议 `REPO_REF` 填 CPU 阶段显示的代码提交 SHA。
3. GPU 笔记本依次恢复环境、预检、生成短片和预览。成功后把 `CONFIG_PATH` 改为 `configs/quality.json` 再生成正式视频。
4. 输出视频与参数记录在 `MyDrive/minimax-h3-colab/outputs`。完成后执行最后一格 `runtime.unassign()`，或使用 Colab「断开连接并删除运行时」。

也可复制下面的完整链接；如果 fork 到其他账号，请同时修改笔记本的 `REPO_URL`：

```text
https://colab.research.google.com/github/simonFelix-Ai/colab/blob/main/notebooks/01_prepare_cpu.ipynb
https://colab.research.google.com/github/simonFelix-Ai/colab/blob/main/notebooks/02_infer_g4.ipynb
```

公开仓库最方便。私有仓库先通过 Colab 的 GitHub 授权方式访问，并为运行时内的 `git clone` 配置相应认证；不要把 token 写进公开 URL、代码或 notebook 输出。

## 资源与费用预期

| 项目 | 当前配置 |
|---|---|
| GPU | 单张 G4 / RTX PRO 6000 Blackwell 96GB；启动时检测实际型号和空闲显存 |
| 权重 | 单任务完整 BF16，共 144,051,160,615 bytes，约 144.05GB / 134.16GiB |
| Drive | 建议至少 180GB **可用**空间，容纳权重、CUDA 安装包和下载余量 |
| GPU 系统 RAM | `staged` 文生视频保守预检要求 85GiB 可用；`auto_offload` 要求 150GiB 可用 |
| CPU 临时磁盘 | 构建及试装 CUDA 轮子也需要空间，建议至少 25GiB 可用 |
| GPU 本地磁盘 | 默认直接读 Drive；若复制权重到 `/content`，额外需要约 140GiB 可用空间 |
| smoke | 960×544，124 帧，24fps，约 5.17 秒，8 次去噪模型预测 |
| quality | 1344×768，124 帧，50 次去噪模型预测；只是较高步数起点，不保证特定成片质量 |

系统 RAM 与 GPU 显存是两种资源。96GB 显存不能证明系统 RAM 足够。上面的 RAM 门槛是本项目的保守预检值，不是实测峰值或官方最低配置。

GPU 会话中的依赖恢复、权重读取、CPU 运算和等待仍可能消耗计算单元。提前准备能减少联网安装、重复下载和现场写代码时间，**不能省掉每台新 GPU 虚拟机的环境恢复与模型加载**。切换硬件可能换虚拟机，`/content` 与 Python 库不能作为持久缓存。具体消耗速度以 Colab 界面为准。

Drive 读取大权重可能很慢，也可能遇到配额。GPU 笔记本提供 `USE_LOCAL_SSD` 选项；是否先复制到本地需要权衡磁盘空间、复制耗时与后续加载速度。

## 文件说明

| 文件 | 用途 |
|---|---|
| `notebooks/01_prepare_cpu.ipynb` | 拉取代码、挂载 Drive、缓存 Linux 环境、CPU API/封装检查、定向下载权重 |
| `notebooks/02_infer_g4.ipynb` | 恢复虚拟环境、检查 GPU/RAM/模型、运行推理、预览、手动释放运行时 |
| `scripts/environment.py` | 构建 wheel 缓存，生成精确 `resolved.txt`，按平台/Python/依赖文件指纹离线恢复 |
| `scripts/h3.py` | 下载计划、定向下载、大小与版本标记检查、资源预检、分阶段加载与推理 |
| `scripts/validate_runtime.py` | 在 CPU 上检查 H3 工作流拆分 API、生成测试 MP4 并检查音视频流 |
| `model-lock.json` | 权重提交、文件名与大小，不包含权重本身 |
| `requirements-colab.txt` | CUDA 12.8 / PyTorch 2.9.1 与固定 Diffusers 源码提交；其余依赖由 CPU 阶段解析并冻结 |
| `configs/` / `prompts/` | 生成参数与可独立修改的提示词 |
| `docs/` | 省计算单元说明、故障排查、来源与验证范围 |

## 修改提示词和参数

本地编辑 `prompts/fox.txt`，提交到 GitHub，再让 Colab 拉取你确认的提交。也可以在配置 JSON 内用 `prompt` 字段替代 `prompt_file`。相对素材/提示词路径以仓库根目录为基准，也支持绝对路径。

首轮保持 `configs/smoke.json` 默认值。544p 只是节省首测时间的较小画布，偏离模型默认训练画布；正式画质参考 768p。

帧数采用 `17*n+5`，且对齐后的时长必须满足当前 Diffusers 实现的 5–15 秒约束：124 帧约 5.17 秒，243 帧约 10.13 秒，345 帧约 14.38 秒。不要直接填 360 或 362 帧，它们会超出此实现的对齐时长范围。

当前固定版本的 `num_inference_steps` **包含末尾零 sigma 点**，因此 9 表示 8 次模型预测、51 表示 50 次。参数不是 SGLang 的同义替换。H3 权重已做 guidance 蒸馏，本项目没有 `negative_prompt` / `guidance_scale` 参数。

图生视频先修改 `configs/image_to_video.json` 的 `image` 为你自己的 Drive 图片绝对路径，可另加 `last_image`。`fl2va` 与 `t2va` 共用下载权重，但图生视频目前走 `auto_offload`，系统 RAM 需求更高。

多模态参考使用 `workflow: ref2va`、`memory_mode: auto_offload`，并增加有序 `references`，例如 `[{"type": "image", "path": "/content/drive/MyDrive/subject.png"}]`。先在 CPU 准备阶段把 `WORKFLOW` 改为 `ref2va` 下载另一套 transformer；两套都保留会多占约 66GB。限制为最多 9 张图片、3 段视频、3 段音频、合计 12 项；音频不能是唯一参考。视频/音频时长限制见官方模型卡；首测建议先用文生视频。

## 命令行入口

以下命令在 Colab 的仓库根目录执行；GPU 运行使用 `.venv/bin/python`，不要用 notebook 内核中碰巧存在的旧依赖。

```bash
python scripts/h3.py plan --workflow t2va
python scripts/environment.py cache --cache /content/drive/MyDrive/minimax-h3-colab/env
python scripts/environment.py install --cache /content/drive/MyDrive/minimax-h3-colab/env
.venv/bin/python scripts/h3.py download --model-dir /content/drive/MyDrive/minimax-h3-colab/models/MiniMax-H3
.venv/bin/python scripts/h3.py check --model-dir /content/drive/MyDrive/minimax-h3-colab/models/MiniMax-H3 --config configs/smoke.json
.venv/bin/python scripts/h3.py run --model-dir /content/drive/MyDrive/minimax-h3-colab/models/MiniMax-H3 --output-dir /content/drive/MyDrive/minimax-h3-colab/outputs --config configs/smoke.json
```

## 上传前检查

Windows 本地不需要安装 PyTorch 或下载模型，即可运行仓库自检：

```powershell
python -m unittest discover -s tests -v
python scripts/h3.py plan
python scripts/h3.py validate-config
```

若希望使用 Git 推送，可在本目录按顺序执行（先在 GitHub 创建空仓库）：

```powershell
git init -b main
git add .
git commit -m "Add MiniMax H3 Colab preparation and inference notebooks"
git remote add origin https://github.com/simonFelix-Ai/colab.git
git push -u origin main
```

## 模型边界

本项目只做开源 H3-Base 的本地推理，不需要 MiniMax API key。官方 H3-Context-IR 和 H3-Regenerate-2K 是单独的托管流程；本项目不会自动调用这些服务。手写提示词的成片效果也不能视为完整官方工作流的复现。

模型使用 MiniMax H3 Community License，区别于推理库的代码许可。使用及公开或商业发布前阅读 [官方权重许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) 与适用条款。

详细资料：[省计算单元说明](docs/cost-and-persistence.md)、[排错](docs/troubleshooting.md)、[来源与验证范围](docs/sources-and-validation.md)。
