# 常见故障

| 现象 | 处理 |
|---|---|
| `YOUR_NAME` 占位符错误 | 修改两份笔记本第一格的 `REPO_URL`，上传整个仓库 |
| 不是 Python 3.12 | 选择兼容 Colab 运行时；若迁移 Python，修改约束并在 CPU 重建缓存，不要跳过指纹检查 |
| `Wheel cache missing` | 确认挂载同一账号的 Drive、两份笔记本路径一致、CPU cache 阶段成功 |
| pip 版本冲突/下载失败 | 留在 CPU 会话，根据完整错误处理约束或网络后重跑；成功前不要分配 G4 |
| CPU 缓存阶段 API 或 MP4 检查失败 | 表明这组依赖与 H3 API 或媒体编码不兼容；缓存不会写完成标记，修复后再进入 GPU |
| Drive 空间不足 / `Input/output error` | 检查 Drive 可用配额与垃圾箱，换有空间的目录；`disk_usage` 对 Drive 的结果不能替代账号配额 |
| 下载中断 | 使用原目录重新执行 CPU download；保留下载元数据，可利用已完成文件 |
| Hugging Face 401/403 | 在模型页查看授权/许可要求，Colab Secrets 添加有读取权限的 `HF_TOKEN`，不要提交 token |
| `Missing/incomplete model files` | 检查路径和 workflow；在 CPU 会话重新 download。ref2va 与 t2va 的 transformer 不同 |
| GPU 检查无 CUDA | 选择 G4，确认分配成功；96GB 系统 RAM 不是 96GB GPU 显存 |
| 可用显存不足 80GiB | 确认真的是 G4 96GB，结束其他模型进程；不要直接用 T4/L4 套此 BF16 配置 |
| `no kernel image` / `sm_120` 不支持 | 确认调用 `.venv/bin/python` 与 CUDA 12.8 PyTorch 轮子，检查驱动；不要临时降级到旧 CUDA torch |
| 系统 RAM 预检失败 | t2va 用 staged；选择更大系统 RAM。图生/参考模式当前需要较大 RAM，减少帧数并不能减少全部权重的常驻大小 |
| `Components failed to load` | 查看之前的 Diffusers 组件警告，通常是缺权重、路径或依赖/API 不兼容；CPU 检查阶段尽早定位 |
| CUDA OOM | 回到 smoke 的 960×544/124 帧，清理其他 GPU 进程；若刚失败可重启干净运行时。不要反复盲试高分辨率长片 |
| OOM 出现在 VAE 解码 | 先减小画布/帧数。当前脚本不假定所有 H3 VAE 都支持相同 tiling API，没有乱调其他模型专用参数 |
| 首次模型加载非常慢 | 约 144GB 大权重经 Drive 读取本来就可能慢；磁盘足够可选本地 SSD 复制，计入复制成本后再比较 |
| 低步数画质差 | smoke 用于联通性检查；通过后提高步数并参考官方提示词指南 |
| 输入图片找不到 | 使用 GPU 运行时能读取的路径，最好是已挂载 Drive 的绝对路径；本机 Windows 路径不可用 |
| 推理报错 | 查看同名输出 JSON 的 `error` 和 notebook traceback；加载/运行失败不会自动反复重试 |
| 预览无声 | 先确认播放器未静音，再用支持 AAC 的播放器检查文件；CPU 准备阶段会检查 MP4 的音频与视频流 |

下载计划和配置校验不需要 torch，也不需要 GPU：

```bash
python scripts/h3.py plan --workflow t2va
python scripts/h3.py validate-config --config configs/smoke.json
```

收集诊断时保留 Python/PyTorch/CUDA 版本、GPU 名称、RAM/显存、代码提交和完整 traceback；不要公开 token 或私人素材。模型加载前资源预检若失败，不会创建成片任务记录，直接查看 notebook 输出。
