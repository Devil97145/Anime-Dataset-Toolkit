# ui/mikan_magnet_ui.py
import gradio as gr
import os
import threading
import subprocess
import time
from pathlib import Path

# 指定本地 aria2c 路径（相对于项目根目录）
ARIA2C_PATH = os.path.join("utils", "aria2-1.37.0-win-64bit-build1", "aria2c.exe")

def open_mikan_page():
    """在浏览器中打开米咔主页"""
    import webbrowser
    webbrowser.open("https://mikanani.kas.pub/")
    return "✅ 已在浏览器中打开蜜柑页面，请点击番剧进入详情页复制磁力链接"


def download_with_aria2(magnet_link, download_dir):
    if not magnet_link or not download_dir:
        return "❌ 请填写磁力链接和下载路径"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)

    # 检查 aria2c 是否存在
    if not os.path.exists(ARIA2C_PATH):
        return f"❌ aria2c 未找到，请确认路径：{ARIA2C_PATH}"

    cmd = [
        ARIA2C_PATH,  # ← 使用本地路径
        "--dir", download_dir,
        "--seed-time=0",
        "--file-allocation=none",
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        magnet_link
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 最长2小时
        if result.returncode == 0:
            # 查找视频文件
            video_files = []
            for ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv']:
                video_files.extend(Path(download_dir).glob(f"*{ext}"))
            if video_files:
                latest = max(video_files, key=os.path.getctime)
                return str(latest)
            else:
                return "⚠️ 下载完成，但未找到视频文件"
        else:
            return f"❌ 下载失败: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "⏰ 下载超时（超过2小时）"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def create_mikan_tab():
    aria2_status = "✅ aria2 就绪" if os.path.exists(ARIA2C_PATH) else "❌ aria2 未安装"

    with gr.TabItem("📺 蜜柑计划 + 磁力下载"):
        gr.Markdown(f"## 📺 蜜柑计划资源站 + 磁力下载一体化\n> {aria2_status}")
        
        with gr.Row():
            open_btn = gr.Button("🌐 打开蜜柑计划页面", variant="primary")
            open_status = gr.Textbox(label="状态", interactive=False, lines=1)

        gr.Markdown("### 🔗 磁力链接下载")
        with gr.Row():
            with gr.Column():
                magnet_input = gr.Textbox(label="🧲 磁力链接", placeholder="粘贴从蜜柑页面复制的 magnet 链接")
                download_dir = gr.Textbox(label="📁 下载路径", value="G:/Downloads", placeholder="如：G:/Downloads")
                download_btn = gr.Button("🔽 开始下载", variant="primary")
            with gr.Column():
                download_status = gr.Textbox(label="📊 下载状态", lines=3, interactive=False)
                video_player = gr.Video(label="🎥 下载完成后自动播放", visible=False)

        # 绑定事件
        open_btn.click(fn=open_mikan_page, inputs=[], outputs=open_status)

        def start_download(magnet, ddir):
            result = [""]
            def run():
                output = download_with_aria2(magnet, ddir)
                result[0] = output
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(timeout=5)  # 短暂等待，避免卡住
            final = result[0]
            if final.endswith(('.mp4', '.mkv', '.avi')):
                return final, gr.update(visible=True, value=final)
            else:
                return final, gr.update(visible=False)

        download_btn.click(
            fn=start_download,
            inputs=[magnet_input, download_dir],
            outputs=[download_status, video_player]
        )