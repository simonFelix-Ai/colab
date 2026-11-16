# 版本、来源与验证范围

资料核对日期：2026-11-16。

## 固定版本

| 组件 | 版本 |
|---|---|
| 权重库 | `MiniMaxAI/MiniMax-H3` |
| 权重 revision | `0919fb684102ef115126612f3fbb84d1027e7317` |
| Diffusers 源码 revision | `f53d552036a0d1bd5570782a39cd40cfabf112bc` |
| PyTorch / torchvision | `2.9.1+cu128` / `0.24.1+cu128` |
| 目标运行时 | Linux x86_64，Python 3.12 |
| 其他依赖 | CPU 阶段按 requirements 约束解析；精确版本写入 Drive `resolved.txt` |

H3 的此版本是 **ModularPipeline** 集成。没有套用 Hugging Face 页面自动生成的通用 `DiffusionPipeline(...).images` 片段，也没有用未固定的 `main` 作为推理库版本。

## 官方资料

- [MiniMax 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3)：本地任务、工作流、托管 2K 边界。
- [官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)：权重、组件布局、输入限制。
- [固定 Diffusers 文档](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/docs/source/en/api/pipelines/minimax_h3.md)：ModularPipeline、ComponentsManager、工作流、模型大小、帧对齐、音视频导出与拆分文本编码器的 API。
- [对应 PR](https://github.com/huggingface/diffusers/pull/14355)：H3 集成背景；本项目使用固定提交，不动态跟随 PR head。
- [PyTorch 历史安装版本](https://pytorch.org/get-started/previous-versions/)：CUDA wheel 安装。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)：运行时生命周期、Drive I/O、动态资源限制。
- [Google Cloud G4 机器类型](https://docs.cloud.google.com/compute/docs/gpus)：RTX PRO 6000 Blackwell 与 96GB GPU 内存；Colab 实际分配仍以预检为准。

## 实现选择

默认仅下载共享组件和 `transformer/`，跳过 `transformer_ref/`、原始 `FL2VA/` 与 `Ref2VA/`，减少重复百 GB 下载。文件清单来自固定 revision 的 Hugging Face 文件元数据。下载通过官方 Hub 客户端，随后检查文件大小与 revision 标记；本项目没有额外遍历读取全部 144GB 做第二遍 SHA256。

默认文生视频采用文档中的拆分 block API，将文本编码阶段与生成阶段放在同一张卡上顺序执行，编码器完全卸载后才加载 transformer 和 VAE。这是对文档拆分示例的单卡适配，**不是官方提供的 Colab G4 实测配方**。图生/参考模式沿用 ComponentsManager 自动 CPU 卸载，不在未验证的媒体状态边界强行拆分。

默认使用 PyTorch 自带 attention 路径，不安装 CUDA 扩展，不在 Blackwell 上直接套用文档中注明适合 Hopper 的 FlashAttention 3 配方。之后可单独基准测试适合 SM120 的优化，但首轮保持依赖尽量简单。

## 验证层次

1. 本地 Windows：检查 Python 语法、notebook JSON/所有代码单元、配置有效性、模型清单过滤、缺文件/错版本拒绝等控制逻辑；无需 GPU 和权重。
2. CPU Colab 准备阶段：真正解析 Linux CUDA 轮子、安装整套环境、运行 `pip check`、导入 H3 类并检查拆分工作流 API、生成短测试 MP4 并检查双流。不加载 H3 大权重，也不做模型推理。
3. GPU Colab：真正检测 CUDA、BF16 与矩阵计算，核对可用显存/RAM和本地权重，再加载并生成 smoke 视频。

交付时仅完成本地可执行检查与上游源码核对，未运行第 2、3 层。实际速度、计算单元、显存峰值和出片效果需要你的 Colab 首次运行确认；未虚构性能数据。
