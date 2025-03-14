import random
import re

def better_augmentation(text):
    augmented_data = []
    # 保留原始文本
    augmented_data.append(text)
    
    # 按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    all_sentences = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # 收集完整句子
        sentences = paragraph.split('\n')
        all_sentences.extend([s for s in sentences if len(s.strip()) > 5])
        
        # 1. 段落保持完整性增强
        if len(sentences) > 2:
            # 交换句子顺序
            for _ in range(2):
                shuffled = sentences.copy()
                random.shuffle(shuffled)
                augmented_data.append('\n'.join(shuffled))
        
        # 2. 句子部分变换（轻微修改）
        for sent in sentences:
            if len(sent) < 10:
                continue
                
            # 添加完整句
            augmented_data.append(sent)
            
            # 句子插入修饰词
            modifiers = ["或许", "也许", "大概", "可能", "似乎", "仿佛"]
            if len(sent) > 15 and " " not in sent[:5]:
                modified = sent[:5] + random.choice(modifiers) + sent[5:]
                augmented_data.append(modified)
    
    # 3. 跨段组合句子
    if len(all_sentences) > 10:
        for _ in range(len(paragraphs) * 2):
            # 随机选择3-5个句子组合
            count = random.randint(3, min(5, len(all_sentences)))
            selected = random.sample(all_sentences, count)
            augmented_data.append('\n'.join(selected))
    
    return '\n\n'.join(filter(None, augmented_data))

def main():
    input_path = "c:/CS-Learning/MachineLearning/my_transformer/data/lyrics_aug.txt"
    output_path = "c:/CS-Learning/MachineLearning/my_transformer/data/lyrics_augmented.txt"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    augmented_text = better_augmentation(text)
    
    # 清理多余空行
    augmented_text = re.sub(r'\n{3,}', '\n\n', augmented_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(augmented_text)
    
    original_lines = len(text.split('\n'))
    augmented_lines = len(augmented_text.split('\n'))
    
    print(f"原始文本行数: {original_lines}")
    print(f"增强后文本行数: {augmented_lines}")
    print(f"增长倍数: {augmented_lines/original_lines:.2f}倍")

if __name__ == "__main__":
    main()