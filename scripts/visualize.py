import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.express as px
import os
from collections import defaultdict

# 新增函数：高频短语发现
def discover_phrases(text, min_count=10, max_length=4):
    """发现高频词语组合"""
    phrase_counts = defaultdict(int)
    
    # 发现不同长度的短语
    for length in range(2, max_length+1):
        # 滑动窗口遍历文本
        for i in range(len(text) - length + 1):
            phrase = text[i:i+length]
            phrase_counts[phrase] += 1
    
    # 过滤低频短语
    phrases = {phrase: count for phrase, count in phrase_counts.items() 
              if count >= min_count and len(phrase) >= 2}
    print(f"发现 {len(phrases)} 个高频短语（长度2-{max_length}，出现次数≥{min_count}）")
    return phrases

# 修改后的数据加载函数
def load_data_and_model():
    """加载数据和模型"""
    # 路径配置
    model_path = "model.pth"
    data_path = "data/lyrics_augmented.txt" 
    chars_path = "data/chars.txt"
    
    # 加载字符集
    with open(chars_path, 'r', encoding='utf-8') as f:
        chars = f.read()
    
    # 创建字符到索引的映射
    c2i = {c:i for i, c in enumerate(chars)}
    i2c = {i:c for i, c in enumerate(chars)}
    
    # 加载模型超参数
    vocab_size = len(chars)
    embedding_dim = 384
    block_size = 128
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LanguageModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 计算词频和短语频率
    word_freqs = defaultdict(int)
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        # 单字频率
        for char in text:
            if char in c2i:
                word_freqs[char] += 1
        # 短语发现
        phrases = discover_phrases(text, min_count=20, max_length=4)
        word_freqs.update(phrases)
    
    return model, chars, c2i, i2c, word_freqs, device

# 新增函数：获取短语嵌入
def get_phrase_embedding(phrase, c2i, embedding_matrix):
    """获取短语的嵌入表示（平均池化）"""
    valid_chars = [c for c in phrase if c in c2i]
    if len(valid_chars) == 0:
        return None
    
    # 获取各字符的嵌入
    char_indices = [c2i[c] for c in valid_chars]
    char_embeddings = embedding_matrix[char_indices]
    
    # 平均池化
    return np.mean(char_embeddings, axis=0)

# 修改后的嵌入提取函数
def extract_embeddings(model, chars, c2i, i2c, word_freqs, top_n=100):
    """提取嵌入向量"""
    # 获取词嵌入矩阵
    embedding_matrix = model.vocab_embedding.weight.detach().cpu().numpy()
    
    # 获取频率最高的N个单字
    single_chars = [(k,v) for k,v in word_freqs.items() if len(k) == 1]
    top_words = sorted(single_chars, key=lambda x: x[1], reverse=True)[:top_n]
    top_word_indices = [c2i[word[0]] for word in top_words]
    top_word_embeddings = embedding_matrix[top_word_indices]
    
    # 获取高频短语的嵌入
    phrases = [(k,v) for k,v in word_freqs.items() if len(k) >= 2]
    top_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)[:top_n]
    phrase_embeddings = []
    valid_phrases = []
    
    for phrase, _ in top_phrases:
        emb = get_phrase_embedding(phrase, c2i, embedding_matrix)
        if emb is not None:
            phrase_embeddings.append(emb)
            valid_phrases.append(phrase)
    
    # 组合单字和短语的嵌入
    combined_embeddings = np.vstack([top_word_embeddings, np.array(phrase_embeddings)])
    combined_indices = top_word_indices + valid_phrases
    
    print(f"总嵌入数量：{len(combined_indices)}（单字 {len(top_word_indices)}，短语 {len(valid_phrases)}）")
    return combined_embeddings, combined_indices, embedding_matrix

# 修改后的可视化函数
def visualize_embeddings_3d(combined_indices, embeddings_3d, i2c, word_freqs):
    """3D可视化"""
    # t-SNE降维
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    embeddings_3d = tsne.fit_transform(embeddings_3d)
    
    # 创建3D散点图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 准备可视化数据
    xs = embeddings_3d[:, 0]
    ys = embeddings_3d[:, 1]
    zs = embeddings_3d[:, 2]
    
    # 获取频率和类型（单字/短语）
    freqs = []
    categories = []
    for idx in combined_indices:
        if isinstance(idx, int):  # 单字
            word = i2c[idx]
            freqs.append(word_freqs[word])
            categories.append(1)
        else:  # 短语
            freqs.append(word_freqs[idx])
            categories.append(len(idx))  # 用长度作为类别
    
    # 正规化大小
    sizes = 30 + 70 * (np.array(freqs) - min(freqs)) / (max(freqs) - min(freqs))
    
    # 根据类型设置颜色和形状
    scatter = ax.scatter(xs, ys, zs, 
                        c=categories, 
                        cmap='tab20',
                        s=sizes, 
                        alpha=0.7,
                        marker='o')
    
    # 添加标签（前20个高频元素）
    sorted_indices = np.argsort(freqs)[::-1][:20]
    for i in sorted_indices:
        label = i2c[combined_indices[i]] if isinstance(combined_indices[i], int) else combined_indices[i]
        ax.text(xs[i], ys[i], zs[i], label, size=10, zorder=10)
    
    # 添加图例
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["单字" if l == "1" else f"{int(l)}字词" for l in labels],
              title="词类型", bbox_to_anchor=(1.05, 1))
    
    ax.set_title('字符与短语嵌入可视化（t-SNE 3D投影）')
    plt.savefig('embedding_viz.png', bbox_inches='tight')
    print("保存可视化结果到 embedding_viz.png")

# 修改后的交互式可视化
def interactive_visualization(combined_indices, embeddings_3d, i2c, word_freqs):
    """交互式3D可视化"""
    # 准备数据
    data = []
    for idx, (x, y, z) in zip(combined_indices, embeddings_3d):
        if isinstance(idx, int):
            word = i2c[idx]
            length = 1
        else:
            word = idx
            length = len(word)
        
        data.append({
            'word': word,
            'frequency': word_freqs[word],
            'length': length,
            'x': x,
            'y': y,
            'z': z
        })
    
    df = pd.DataFrame(data)
    
    # 创建可视化
    fig = px.scatter_3d(df, 
                        x='x', y='y', z='z',
                        color='frequency',
                        size='frequency',
                        hover_name='word',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        symbol='length',
                        title='字符与短语嵌入可视化')
    
    fig.update_traces(marker=dict(opacity=0.7))
    fig.write_html('interactive_viz.html')
    print("保存交互式可视化到 interactive_viz.html")

# 新增函数：查找相似词语
def find_similar_phrases(phrase, c2i, embedding_matrix, top_k=10):
    """查找相似词语"""
    emb = get_phrase_embedding(phrase, c2i, embedding_matrix)
    if emb is None:
        print(f"短语 '{phrase}' 包含未知字符")
        return
    
    # 计算余弦相似度
    similarities = []
    for idx, word_emb in enumerate(embedding_matrix):
        cos_sim = np.dot(emb, word_emb) / (np.linalg.norm(emb)*np.linalg.norm(word_emb))
        similarities.append((idx, cos_sim))
    
    # 排序并过滤
    sorted_sims = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k+2]
    
    print(f"与 '{phrase}' 最相似的词语：")
    for idx, score in sorted_sims:
        if idx < len(i2c):  # 单字
            print(f"单字 '{i2c[idx]}': {score:.3f}")
        else:  # 短语
            pass

def main():
    # 加载数据和模型
    model, chars, c2i, i2c, word_freqs, device = load_data_and_model()
    
    # 提取嵌入
    embeddings, combined_indices, embedding_matrix = extract_embeddings(
        model, chars, c2i, i2c, word_freqs, top_n=100
    )
    
    # 可视化
    visualize_embeddings_3d(combined_indices, embeddings, i2c, word_freqs)
    interactive_visualization(combined_indices, embeddings, i2c, word_freqs)
    
    # 示例查询
    test_phrases = ["夏天", "回忆", "爱情"]
    for phrase in test_phrases:
        if all(c in c2i for c in phrase):
            print(f"\n分析短语 '{phrase}':")
            emb = get_phrase_embedding(phrase, c2i, embedding_matrix)
            find_similar_phrases(phrase, c2i, embedding_matrix)
    
if __name__ == "__main__":
    main()