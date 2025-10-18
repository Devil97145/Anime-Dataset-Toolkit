# ui/filtering_ui.py
import gradio as gr
import threading
import os
from functions.txt_image_processor import TxtWithImageProcessor, get_top_keywords_from_dir

def append_keyword(keyword_input, keyword_to_add):
    if " (" in keyword_to_add:
        keyword_clean = keyword_to_add.split(' (')[0].strip()
    else:
        keyword_clean = keyword_to_add.strip()
    if not keyword_input.strip():
        return keyword_clean
    current = [k.strip() for k in keyword_input.split(',') if k.strip()]
    if keyword_clean not in current:
        current.append(keyword_clean)
    return ", ".join(current)

def analyze_top_keywords(source_dir):
    if not source_dir or not os.path.exists(source_dir):
        return [], "❌ 源文件夹不存在！"
    try:
        top_words = get_top_keywords_from_dir(source_dir, top_n=20)
        if not top_words:
            return [], "⚠️ 未找到TXT文件或无有效词"
        pure = [w for w, c in top_words]
        total = sum(c for _, c in top_words)
        info = f"✅ 分析完成！共 {len(pure)} 个高频词（{total} 次出现）"
        return pure, info
    except Exception as e:
        return [], f"❌ 分析失败：{str(e)}"

def process_txt_images(source_dir, keyword, target_dir, case_sensitive, fuzzy_match, max_threads, action_choice, match_mode):
    if not source_dir or not keyword:
        return "❌ 请填写源文件夹和关键词！"
    action = "delete" if action_choice == "永久删除文件" else "move"
    if action == "move" and (not target_dir or not os.path.exists(target_dir)):
        return f"❌ 目标文件夹不存在：{target_dir}"
    mode_map = {"任一匹配（OR）": "or", "全部匹配（AND）": "and"}
    match_mode_en = mode_map.get(match_mode, "or")
    processor = TxtWithImageProcessor(
        source_dir=source_dir,
        keyword=keyword,
        action=action,
        target_dir=target_dir if action == "move" else None,
        case_sensitive=case_sensitive,
        fuzzy_match=fuzzy_match,
        max_threads=int(max_threads),
        match_mode=match_mode_en
    )
    processor.start()
    verb = "删除" if action == "delete" else "移动"
    return f"✅ TXT+图片{verb}处理完成！"

def create_filtering_tab():
    with gr.TabItem("📄 智能筛选"):
        gr.Markdown("## 🔍 智能标签筛选")
        with gr.Row():
            with gr.Column(scale=2):
                source_dir2 = gr.Textbox(label="📂 源文件夹路径")
                gr.Markdown("### 📊 高频标签分析")
                with gr.Row():
                    analyze_btn = gr.Button("📊 分析高频标签（前20）")
                    analyze_info = gr.Textbox(label="分析结果", interactive=False, lines=1)
                gr.Markdown("### 🔖 高频标签快捷选择")
                top_kw_buttons = []
                for i in range(4):
                    with gr.Row():
                        for j in range(5):
                            btn = gr.Button(f"标签{i*5+j+1}", visible=False, size="sm")
                            top_kw_buttons.append(btn)
                gr.Markdown("### 🔑 匹配设置")
                keyword2 = gr.Textbox(label="🔑 匹配标签")
                with gr.Row():
                    match_mode = gr.Radio(["任一匹配（OR）", "全部匹配（AND）"], value="任一匹配（OR）")
                    action_choice = gr.Radio(["移动到目标文件夹", "永久删除文件"], value="移动到目标文件夹")
                target_dir2 = gr.Textbox(label="📁 目标文件夹路径（仅移动时需要）")
                with gr.Accordion("高级选项", open=False):
                    with gr.Row():
                        case_sensitive = gr.Checkbox(False, label="🔠 区分大小写")
                        fuzzy_match = gr.Checkbox(False, label="🌀 模糊匹配")
                        max_threads2 = gr.Slider(1, 16, 4, step=1, label="🧵 最大线程数")
                btn2 = gr.Button("🔍 开始处理", variant="primary", size="lg")
            with gr.Column(scale=1):
                output2 = gr.Textbox(label="📊 处理结果", lines=10)

        top_words_state = gr.State([])

        analyze_btn.click(analyze_top_keywords, [source_dir2], [top_words_state, analyze_info])

        def update_buttons(words):
            updates = []
            for i in range(20):
                if i < len(words):
                    updates.append(gr.update(value=words[i], visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        top_words_state.change(update_buttons, [top_words_state], top_kw_buttons)

        for btn in top_kw_buttons:
            btn.click(append_keyword, [keyword2, btn], keyword2)

        btn2.click(
            process_txt_images,
            [source_dir2, keyword2, target_dir2, case_sensitive, fuzzy_match, max_threads2, action_choice, match_mode],
            output2
        )