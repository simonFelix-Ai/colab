"""Regenerate the two Colab notebooks with the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def notebook(name, cells, gpu=False):
    target = ROOT / "notebooks" / name
    target.parent.mkdir(exist_ok=True)
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha256((name + ":" + str(i)).encode()).hexdigest()[:12]
    metadata = {"colab": {"name": name, "provenance": [], "private_outputs": True},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"}}
    if gpu:
        metadata["accelerator"] = "GPU"
    target.write_text(json.dumps({"cells": cells, "metadata": metadata,
                                 "nbformat": 4, "nbformat_minor": 5}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


CONFIG = '''
# 已填写当前仓库地址；如 fork 到其他账号，请修改此项。
REPO_URL = "https://github.com/simonFelix-Ai/colab.git"
REPO_REF = "main"  # 建议确认后改成自己的提交 SHA
PERSIST_ROOT = "/content/drive/MyDrive/minimax-h3-colab"
WORKFLOW = "t2va"  # t2va / fl2va 共用权重；ref2va 需要另一个 transformer
'''

CLONE = '''
import os, subprocess, sys
from pathlib import Path
assert REPO_URL.startswith("https://github.com/"), "请填写有效的 GitHub 仓库 URL"
assert sys.version_info[:2] == (3, 12), "本项目使用 Python 3.12，请切到兼容的 Colab 运行时"
PROJECT = Path("/content/minimax-h3-colab")
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(PROJECT)], check=True)
else:
    assert subprocess.check_output(["git", "-C", str(PROJECT), "remote", "get-url", "origin"], text=True).strip() == REPO_URL, "现有目录属于其他仓库"
subprocess.run(["git", "-C", str(PROJECT), "fetch", "origin", REPO_REF], check=True)
subprocess.run(["git", "-C", str(PROJECT), "checkout", "--detach", "FETCH_HEAD"], check=True)
os.chdir(PROJECT)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("代码版本:", COMMIT)
def run(*args):
    subprocess.run([str(x) for x in args], check=True, cwd=PROJECT)
'''

DRIVE = '''
from google.colab import drive
drive.mount("/content/drive")
PERSIST = Path(PERSIST_ROOT)
PERSIST.mkdir(parents=True, exist_ok=True)
import hashlib
ENV_KEY = hashlib.sha256((PROJECT / "requirements-colab.txt").read_bytes()).hexdigest()[:12]
ENV_CACHE = PERSIST / ("environment-py312-" + ENV_KEY)
MODEL_DIR = PERSIST / "models/MiniMax-H3"
PY = PROJECT / ".venv/bin/python"
'''

TOKEN = '''
# 可选：Colab 左侧钥匙图标中新建 HF_TOKEN，并授权当前笔记本。
# 不要把 token 粘贴进代码、配置或 GitHub。公开权重可不填。
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
except Exception:
    pass
'''


def main():
    notebook("01_prepare_cpu.ipynb", [
        md("""# MiniMax H3：CPU 提前准备
先选 **运行时 → 更改运行时类型 → CPU / 无硬件加速**，Python 3.12。这里不需要 GPU。
本笔记本将 Linux CUDA 安装包、精确依赖清单和模型权重保存在你的 Google Drive。
单任务权重约 **144 GB（134.2 GiB）**，另需安装包和余量，建议 Drive 可用空间 **180 GB 以上**。
CPU 会话可以长时间下载；中断后重新执行可利用 Hugging Face 已完成的缓存。
不要假设切换 GPU 后 `/content` 和已安装的库仍在。"""),
        code(CONFIG), code(CLONE), code(DRIVE), code(TOKEN),
        md("## 1. 查看下载清单\n这一步只读随仓库提交的模型清单，无须安装依赖。"),
        code('run(sys.executable, "scripts/h3.py", "plan", "--workflow", WORKFLOW)'),
        md("## 2. 缓存环境\n下载 CUDA 12.8 的 PyTorch 轮子，并构建固定提交的 Diffusers；不会编译 CUDA 内核。随后在 CPU 上检查 H3 API 和 MP4 音画封装。首次耗时较长；完成标记存在时跳过重复构建。"),
        code('''
if not (ENV_CACHE / "environment.json").is_file():
    run(sys.executable, "scripts/environment.py", "cache", "--cache", ENV_CACHE)
else:
    print("已有环境缓存；如需重新解析依赖，请显式重跑 cache 命令。")
run(sys.executable, "scripts/environment.py", "install", "--cache", ENV_CACHE)
'''),
        md("## 3. 下载并检查模型\n只下载选定任务的 Diffusers 组件，不下载另一套 transformer 或原始 FL2VA/Ref2VA 副本。先确认已阅读模型许可；若 Hugging Face 要求授权，请先在模型页完成。"),
        code('''
run(PY, "scripts/h3.py", "download", "--workflow", WORKFLOW, "--model-dir", MODEL_DIR)
run(PY, "scripts/h3.py", "verify", "--workflow", WORKFLOW, "--model-dir", MODEL_DIR)
'''),
        md("## 4. 准备完成\n记下上方代码提交 SHA。结束 CPU 会话，在 GPU 笔记本设置相同仓库、提交和 Drive 路径。\n配置和提示词尽量在本地修改后推到 GitHub。权重、安装包、token 和私人输入素材不要上传 GitHub。"),
        code('print("准备完成。代码提交:", COMMIT, "\\nDrive:", PERSIST)')])

    notebook("02_infer_g4.ipynb", [
        md("""# MiniMax H3：G4 96GB 本地推理
先完成 `01_prepare_cpu.ipynb`。选择 **G4 GPU**；查看系统 RAM，默认 staged 模式预检要求至少 **85 GiB 可用 RAM**。
这里会重新拉取代码并用 Drive 轮子离线恢复环境，随后做 GPU 检查和短片推理。
GPU 分配期间安装、读取权重、等待等也可能消耗计算单元；本方案减少准备和调试耗时，不能让启动开销归零。
默认不调用 MiniMax 付费 API；2K 服务不在此本地流程内。"""),
        code(CONFIG), code(CLONE), code(DRIVE),
        md("## 1. 离线恢复环境\n没有缓存时立即停止，返回 CPU 笔记本准备。独立虚拟环境通过子进程使用，无须重启 notebook 内核。"),
        code('''
run(sys.executable, "scripts/environment.py", "install", "--cache", ENV_CACHE)
run(PY, "scripts/validate_runtime.py")
'''),
        md("## 2. 设置生成配置\n首次用 smoke。staged 文生视频先编码提示词，再释放编码器，然后加载视频模型，减少 RAM 峰值。\nimage_to_video 使用 auto_offload，预检要求至少 150 GiB 可用 RAM；先准备好图片路径。\n默认直接读取 Drive 权重。可选复制到本地 SSD，需额外约 140 GiB 空闲磁盘，复制本身也占 GPU 会话时间。"),
        code('''
CONFIG_PATH = PROJECT / "configs/smoke.json"  # 成功后改 configs/quality.json
OUTPUT_DIR = PERSIST / "outputs"
USE_LOCAL_SSD = False
if USE_LOCAL_SSD:
    import shutil
    local_model = Path("/content/h3-model")
    need = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
    assert shutil.disk_usage("/content").free > need + 10 * 1024**3, "本地磁盘空间不足"
    shutil.copytree(MODEL_DIR, local_model, dirs_exist_ok=True)
    MODEL_DIR = local_model
run(PY, "scripts/h3.py", "validate-config", "--config", CONFIG_PATH)
run(PY, "scripts/h3.py", "check", "--model-dir", MODEL_DIR, "--config", CONFIG_PATH)
'''),
        md("## 3. 生成视频和原生音频\n124 帧约 5.17 秒。smoke 的 9 个 sigma 点执行 8 次模型预测，主要验证流程；正式配置用更多步数。输出直接保存到 Drive，附 JSON 参数、耗时、显存与错误记录。每次点击都会新建一条生成任务。"),
        code('run(PY, "scripts/h3.py", "run", "--model-dir", MODEL_DIR, "--output-dir", OUTPUT_DIR, "--config", CONFIG_PATH)'),
        md("## 4. 预览与结束\n确认输出保存在 Drive 后执行最后一格释放运行时；不会自动重试或在后台循环生成。"),
        code('''
from IPython.display import Video, display
videos = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
assert videos, "还没有生成 MP4"
display(Video(str(videos[-1]), embed=True))
print(videos[-1])
'''),
        code('''
# 完成后手动执行这一格，结束当前运行时。
from google.colab import runtime
runtime.unassign()
''')], gpu=True)


if __name__ == "__main__":
    main()
