# functions/image_tag_editor.py
import os
import glob
import re  # ← 新增导入
from PIL import Image

class ImageTagEditor:
    def __init__(self):
        self.current_image_path = None
        self.current_txt_path = None
        self.image_list = []
        self.current_index = 0
        self.histories = {}
        self.history_indices = {}

    def _switch_history(self, txt_path):
        if txt_path not in self.histories:
            self.histories[txt_path] = []
            self.history_indices[txt_path] = -1

    def load_images(self, folder_path):
        if not folder_path or not os.path.exists(folder_path):
            return [], None, "", "❌ 文件夹路径无效或不存在"

        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files = sorted(image_files)
        self.image_list = image_files
        self.current_index = 0

        if not image_files:
            return [], None, "", "⚠️ 未找到任何图像文件"

        first_image = image_files[0]
        img, tags, msg = self.load_image_and_tags(first_image)
        self.current_image_path = first_image
        self.current_txt_path = first_image.rsplit('.', 1)[0] + '.txt'
        self._switch_history(self.current_txt_path)

        return image_files, img, tags, f"✅ 成功加载 {len(image_files)} 张图像"

    def load_image_and_tags(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            txt_path = image_path.rsplit('.', 1)[0] + '.txt'
            tags = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
            return img, tags, "✅ 图像和标签加载成功"
        except Exception as e:
            return None, "", f"❌ 加载失败: {str(e)}"

    def _normalize_tags(self, tags_str):
        # 支持中英文逗号、分号、换行等分隔符，但输出统一用英文逗号
        separators = r'[,\，;；\n]+'
        tags = [t.strip() for t in re.split(separators, tags_str) if t.strip()]
        return ", ".join(tags)  # 统一用英文逗号保存

    def save_tags(self, tags):
        if not self.current_txt_path:
            return "⚠️ 未加载任何图像，无法保存"
        try:
            old_tags = ""
            if os.path.exists(self.current_txt_path):
                with open(self.current_txt_path, 'r', encoding='utf-8') as f:
                    old_tags = f.read().strip()
            new_tags = self._normalize_tags(tags)
            with open(self.current_txt_path, 'w', encoding='utf-8') as f:
                f.write(new_tags)
            self._add_to_history(old_tags, new_tags)
            return f"✅ 标签已保存到 {os.path.basename(self.current_txt_path)}"
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"

    def _add_to_history(self, old_tags, new_tags):
        hist = self.histories[self.current_txt_path]
        idx = self.history_indices[self.current_txt_path]
        if idx < len(hist) - 1:
            self.histories[self.current_txt_path] = hist[:idx+1]
        self.histories[self.current_txt_path].append((old_tags, new_tags))
        self.history_indices[self.current_txt_path] += 1

    def undo(self):
        if not self.current_txt_path:
            return None, "", "⚠️ 未加载图像"
        idx = self.history_indices[self.current_txt_path]
        if idx < 0:
            return None, "", "⚠️ 无操作可撤销"
        old_tags, _ = self.histories[self.current_txt_path][idx]
        with open(self.current_txt_path, 'w', encoding='utf-8') as f:
            f.write(old_tags)
        self.history_indices[self.current_txt_path] -= 1
        return None, old_tags, "✅ 已撤销上一步操作"

    def redo(self):
        if not self.current_txt_path:
            return None, "", "⚠️ 未加载图像"
        hist = self.histories[self.current_txt_path]
        idx = self.history_indices[self.current_txt_path]
        if idx >= len(hist) - 1:
            return None, "", "⚠️ 无操作可重做"
        self.history_indices[self.current_txt_path] += 1
        _, new_tags = hist[self.history_indices[self.current_txt_path]]
        with open(self.current_txt_path, 'w', encoding='utf-8') as f:
            f.write(new_tags)
        return None, new_tags, "✅ 已重做操作"

    def navigate_image(self, direction, manual_index=None):
        if not self.image_list:
            return None, "", "⚠️ 未加载图像列表"
        if manual_index is not None:
            if 0 <= manual_index < len(self.image_list):
                self.current_index = manual_index
            else:
                return None, "", "❌ 索引超出范围"
        else:
            if direction == "prev":
                self.current_index = (self.current_index - 1) % len(self.image_list)
            elif direction == "next":
                self.current_index = (self.current_index + 1) % len(self.image_list)

        current_image_path = self.image_list[self.current_index]
        img, tags, msg = self.load_image_and_tags(current_image_path)
        self.current_image_path = current_image_path
        self.current_txt_path = current_image_path.rsplit('.', 1)[0] + '.txt'
        self._switch_history(self.current_txt_path)
        progress = f"📄 {self.current_index + 1} / {len(self.image_list)}"
        return img, tags, f"{msg} | {progress}"

    # ========== 批量操作：同样需支持中文逗号 ==========
    def batch_replace_tags(self, folder_path, old_tag, new_tag, use_regex=False):
        if not folder_path or not old_tag:
            return "❌ 请填写文件夹路径和要替换的标签"
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if not txt_files:
            return "⚠️ 未找到任何 .txt 文件"
        count = 0
        import re as regex_mod
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if use_regex:
                    new_content = regex_mod.sub(old_tag, new_tag, content)
                else:
                    new_content = content.replace(old_tag, new_tag)
                if new_content != content:
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    count += 1
            except Exception as e:
                print(f"替换失败 {txt_file}: {e}")
        return f"✅ 成功替换 {count} 个文件中的标签"

    def batch_delete_tags(self, folder_path, tag_to_delete, use_regex=False):
        if not folder_path or not tag_to_delete:
            return "❌ 请填写文件夹路径和要删除的标签"
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if not txt_files:
            return "⚠️ 未找到任何 .txt 文件"
        count = 0
        import re as regex_mod
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # ✅ 支持中英文逗号分割
                tags = [t.strip() for t in re.split(r'[,\，;；\n]+', content) if t.strip()]
                original_len = len(tags)
                if use_regex:
                    pattern = regex_mod.compile(tag_to_delete)
                    tags = [t for t in tags if not pattern.search(t)]
                else:
                    tags = [t for t in tags if t != tag_to_delete.strip()]
                if len(tags) != original_len:
                    new_content = ", ".join(tags)  # 输出统一英文逗号
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    count += 1
            except Exception as e:
                print(f"删除失败 {txt_file}: {e}")
        return f"✅ 成功从 {count} 个文件中删除标签"

    def batch_append_tags(self, folder_path, tags_to_append, position="end"):
        if not folder_path or not tags_to_append:
            return "❌ 请填写文件夹路径和要追加的标签"
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if not txt_files:
            return "⚠️ 未找到任何 .txt 文件"
        # ✅ 输入的追加标签也支持中英文逗号
        append_list = [t.strip() for t in re.split(r'[,\，;；\n]+', tags_to_append) if t.strip()]
        if not append_list:
            return "❌ 追加标签为空"
        count = 0
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # ✅ 读取时支持中英文逗号
                tags = [t.strip() for t in re.split(r'[,\，;；\n]+', content) if t.strip()]
                if position == "start":
                    tags = append_list + tags
                else:
                    tags.extend(append_list)
                seen = set()
                unique_tags = []
                for t in tags:
                    if t not in seen:
                        seen.add(t)
                        unique_tags.append(t)
                new_content = ", ".join(unique_tags)  # 输出统一英文逗号
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1
            except Exception as e:
                print(f"追加失败 {txt_file}: {e}")
        return f"✅ 成功向 {count} 个文件追加标签"