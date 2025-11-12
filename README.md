# 🚀 AI 媒体处理工具箱 / AI Media Processing Toolkit

### 📌 简介
这是一个面向动漫数据集构建的 **一体化本地工具箱**，支持：

- 📺 **蜜柑计划磁力下载**（集成 aria2）
- 🎬 **视频抽帧**（跳过黑屏，支持 GPU 加速）
- 🎞️ **高精度 OP/ED 识别**（实验性）
- 🏷️ **AI 自动打标**（基于 WD-ViT-TAGGER-v3，支持 GPU/CPU）
- 🖼️ **人工标签编辑**（保留标签库、撤销、批量操作）
- 🔍 **智能标签筛选**（OR/AND 模式、高频词分析）
- 🖼️→📝 **AI 图像智能描述**（基于通义千问 Qwen-VL 系列模型，支持单图/批量生成中文描述，输出格式：txt / csv / json）

---

### 🛠️ 安装依赖
```bash
git clone https://github.com/Devil97145/Anime-Dataset-Toolkit.git
cd Anime-Dataset-Toolkit
pip install -r requirements.txt

⚙️ 系统要求
操作系统：Windows / Linux（推荐 Windows）
FFmpeg：必须安装并加入系统 PATH
GPU（可选）：NVIDIA 显卡 + CUDA 驱动（用于加速抽帧和打标）
API 密钥（可选）：使用 Qwen-VL 云端模型需配置 DASHSCOPE_API_KEY 环境变量

▶️ 启动应用
```bash
python app.py
默认访问：http://127.0.0.1:7860

### ✅ 更新说明
- 新增 **图像智能描述** 模块，支持通义千问多模态模型（如 `qwen3-vl-plus` 等）
- 支持单图测试与批量处理，可自定义提示词，输出格式灵活（txt/csv/json）
- 兼容中文路径与中文编码（CSV 使用 UTF-8-BOM 确保 Excel 正常显示）
