# ui/editor_ui.py
import gradio as gr
import os
from functions.image_tag_editor import ImageTagEditor

editor = ImageTagEditor()

def load_folder(folder_path):
    return editor.load_images(folder_path)

def save_current_tags(tags):
    return editor.save_tags(tags)

def go_prev():
    return editor.navigate_image("prev")

def go_next():
    return editor.navigate_image("next")

def go_to_index(index):
    if index is None:
        return None, "", "⚠️ 请输入有效数字"
    return editor.navigate_image(None, int(index) - 1)

def undo_tags():
    return editor.undo()

def redo_tags():
    return editor.redo()

def batch_replace(folder, old, new, regex):
    return editor.batch_replace_tags(folder, old, new, regex)

def batch_delete(folder, tag, regex):
    return editor.batch_delete_tags(folder, tag, regex)

def batch_append(folder, tags, pos):
    return editor.batch_append_tags(folder, tags, "start" if pos == "开头追加" else "end")

def prepend_reserved_tag(current_tags, tag_to_add):
    if not tag_to_add.strip():
        return current_tags
    tag_clean = tag_to_add.strip()
    current_list = [t.strip() for t in current_tags.split(",") if t.strip()]
    if tag_clean in current_list:
        current_list.remove(tag_clean)
    current_list.insert(0, tag_clean)
    return ", ".join(current_list)

def parse_reserved_tags(tags_str):
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]

def update_reserved_buttons(tags_list):
    updates = []
    for i in range(20):
        if i < len(tags_list):
            updates.append(gr.update(value=tags_list[i], visible=True))
        else:
            updates.append(gr.update(visible=False))
    return updates

def create_editor_tab():
    with gr.TabItem("🖼️ 图像标签编辑"):
        gr.Markdown("## 🖼️ 图像标签编辑器")
        with gr.Row():
            with gr.Column(scale=1):
                editor_folder = gr.Textbox(label="📂 图像文件夹路径")
                load_btn = gr.Button("📂 加载图像", variant="primary")
                gr.Markdown("### 🔄 导航")
                with gr.Row():
                    prev_btn = gr.Button("⬅️ 上一张")
                    next_btn = gr.Button("➡️ 下一张")
                with gr.Row():
                    index_input = gr.Number(label="跳转到第几张", value=1, precision=0)
                    go_btn = gr.Button("↪️ 跳转")
                gr.Markdown("### 🧷 编辑操作")
                with gr.Row():
                    undo_btn = gr.Button("↩️ 撤销")
                    redo_btn = gr.Button("↪️ 重做")
                save_btn = gr.Button("💾 保存当前标签", variant="primary")
                gr.Markdown("### 📚 保留标签库")
                reserved_tags_input = gr.Textbox(label="📌 输入常用保留标签（逗号分隔）")
                reserved_tags_state = gr.State([])
                reserved_tag_buttons = []
                for i in range(4):
                    with gr.Row():
                        for j in range(5):
                            btn = gr.Button(f"标签{i*5+j+1}", visible=False)
                            reserved_tag_buttons.append(btn)
                with gr.Accordion("🛠️ 批量操作", open=False):
                    batch_folder = gr.Textbox(label="📁 批量操作文件夹")
                    with gr.Tab("替换标签"):
                        batch_old_tag = gr.Textbox(label="旧标签")
                        batch_new_tag = gr.Textbox(label="新标签")
                        batch_use_regex = gr.Checkbox(label="使用正则表达式")
                        batch_replace_btn = gr.Button("🔄 批量替换")
                    with gr.Tab("删除标签"):
                        batch_del_tag = gr.Textbox(label="要删除的标签")
                        batch_del_regex = gr.Checkbox(label="使用正则表达式")
                        batch_del_btn = gr.Button("🗑️ 批量删除")
                    with gr.Tab("追加标签"):
                        batch_append_tags = gr.Textbox(label="要追加的标签（逗号分隔）")
                        batch_append_pos = gr.Radio(["开头追加", "末尾追加"], value="末尾追加")
                        batch_append_btn = gr.Button("➕ 批量追加")
                    batch_output = gr.Textbox(label="批量操作结果")
            with gr.Column(scale=2):
                image_display = gr.Image(label="🖼️ 当前图像", height=512)
                tag_editor = gr.Textbox(label="📝 标签编辑器（逗号分隔）", lines=5)
                status_display = gr.Textbox(label="ℹ️ 状态信息", interactive=False, lines=1)

        load_btn.click(load_folder, [editor_folder], [gr.State(), image_display, tag_editor, status_display])
        save_btn.click(save_current_tags, [tag_editor], [status_display])
        prev_btn.click(go_prev, [], [image_display, tag_editor, status_display])
        next_btn.click(go_next, [], [image_display, tag_editor, status_display])
        go_btn.click(go_to_index, [index_input], [image_display, tag_editor, status_display])
        undo_btn.click(undo_tags, [], [image_display, tag_editor, status_display])
        redo_btn.click(redo_tags, [], [image_display, tag_editor, status_display])
        batch_replace_btn.click(batch_replace, [batch_folder, batch_old_tag, batch_new_tag, batch_use_regex], [batch_output])
        batch_del_btn.click(batch_delete, [batch_folder, batch_del_tag, batch_del_regex], [batch_output])
        batch_append_btn.click(batch_append, [batch_folder, batch_append_tags, batch_append_pos], [batch_output])
        reserved_tags_input.change(parse_reserved_tags, [reserved_tags_input], [reserved_tags_state])
        reserved_tags_state.change(update_reserved_buttons, [reserved_tags_state], reserved_tag_buttons)
        for btn in reserved_tag_buttons:
            btn.click(prepend_reserved_tag, [tag_editor, btn], [tag_editor])