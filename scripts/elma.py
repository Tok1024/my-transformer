import re
from opencc import OpenCC
import os

def process_lyrics_file(input_file_path, output_file_path):
    cc = OpenCC('t2s')  # 繁体转简体转换器
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 繁体转简体
    simplified_content = cc.convert(content)
    
    # 处理空行：将多个空行替换为一个空行
    processed_content = re.sub(r'\n+', '\n', simplified_content)
    
    lines = processed_content.split('\n')
    final_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            final_lines.append('')
            continue
        # 去除行尾标点
        cleaned_line = re.sub(r'[。，！？：,!.?:]+$', '', line)
        final_lines.append(cleaned_line)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_lines))


if __name__ == "__main__":
    # 修改为你的输入文件路径
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对于项目根目录的路径
    project_root = os.path.dirname(current_dir)  # 返回上一级目录
    input_file_path = os.path.join(project_root, 'data', 'lyrics_augmented.txt')
    output_file_path = os.path.join(project_root, 'data', 'lyrics_augmented.txt')
    process_lyrics_file(input_file_path, output_file_path)
    print(f"处理完成，结果已保存到 {output_file_path}")