import requests
import re
import jieba

# 搜索歌曲的函数
def search_songs(artist_name):
    url = f"https://music.163.com/api/search/get/web?csrf_token=&s={artist_name}&type=1&offset=0&total=true&limit=50"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        songs = data['result']['songs']
        song_ids = []
        for song in songs:
            song_ids.append(song['id'])
        return song_ids
    return []

# 获取中文翻译歌词的函数
def get_translated_lyrics(song_id):
    url = f"https://music.163.com/api/song/lyric?id={song_id}&lv=-1&kv=-1&tv=-1"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # 检查是否存在中文翻译歌词
        if 'tlyric' in data and 'lyric' in data['tlyric']:
            lyric = data['tlyric']['lyric']
            # 去除时间戳
            lyric = re.sub(r'\[.*?\]', '', lyric)
            return lyric.strip()
    return None

# 清理文本函数
def clean_text(text):
    # 去除多余空格和空行
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(non_empty_lines)
    # 去除多余空格，但保留换行符
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除特殊符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n，。！？：,!.?:❀ ]', '', text)
    return text

# 主函数
def main():
    artists = ['ヨルシカ', 'n-buna']
    all_lyrics = []
    for artist in artists:
        song_ids = search_songs(artist)
        for song_id in song_ids:
            lyrics = get_translated_lyrics(song_id)
            if lyrics:
                # 清理文本
                clean_lyric = clean_text(lyrics)
                # 按行处理中文分词
                lines = clean_lyric.split('\n')
                processed_lines = []
                for line in lines:
                    if line.strip():
                        words = jieba.lcut(line)
                        processed_lines.append(''.join(words))
                    else:
                        processed_lines.append('')
                processed_lyric = '\n'.join(processed_lines)
                all_lyrics.append(processed_lyric)

    # 将所有歌词合并为一个字符串
    combined_lyrics = '\n\n'.join(all_lyrics)

    # 保存为文本文件
    with open('lyrics.txt', 'w', encoding='utf-8') as f:
        f.write(combined_lyrics)

if __name__ == "__main__":
    main()