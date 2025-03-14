import requests
import re
import json
from bs4 import BeautifulSoup

# QQ 音乐搜索歌曲
def qqmusic_search(artist_name):
    url = f"https://c.y.qq.com/soso/fcgi-bin/client_search_cp?p=1&n=50&w={artist_name}&format=json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'https://y.qq.com/'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        songs = data['data']['song']['list']
        song_mids = []
        for song in songs:
            song_mids.append(song['songmid'])
        return song_mids
    return []

# QQ 音乐获取歌词
def qqmusic_get_lyrics(song_mid):
    url = f"https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_yqq.fcg?nobase64=1&musicid=0&songmid={song_mid}&uin=0&format=json&platform=yqq"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'https://y.qq.com/'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'lyric' in data:
            lyric = data['lyric']
            # 去除时间戳
            lyric = re.sub(r'\[.*?\]', '', lyric)
            return lyric.strip()
    return None

# 网易云音乐搜索歌曲
def netease_search(artist_name):
    url = f"https://music.163.com/api/search/get/web?csrf_token=&s={artist_name}&type=1&offset=0&total=true&limit=50"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        songs = data['result']['songs']
        song_ids = []
        for song in songs:
            song_ids.append(song['id'])
        return song_ids
    return []

# 网易云音乐获取歌词
def netease_get_lyrics(song_id):
    url = f"https://music.163.com/api/song/lyric?id={song_id}&lv=-1&kv=-1&tv=-1"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'tlyric' in data and 'lyric' in data['tlyric']:
            lyric = data['tlyric']['lyric']
            # 去除时间戳
            lyric = re.sub(r'\[.*?\]', '', lyric)
            return lyric.strip()
    return None

# 酷狗音乐搜索歌曲
def kugou_search(artist_name):
    url = f"https://songsearch.kugou.com/song_search_v2?keyword={artist_name}&page=1&pagesize=50&userid=-1&clientver=&platform=WebFilter&tag=em&filter=2&iscorrection=1&privilege_filter=0"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        songs = data['data']['lists']
        song_hashes = []
        for song in songs:
            song_hashes.append(song['FileHash'])
        return song_hashes
    return []

# 酷狗音乐获取歌词
def kugou_get_lyrics(song_hash):
    url = f"https://lyrics.kugou.com/search?ver=1&man=yes&client=pc&keyword={song_hash}&duration=0&hash={song_hash}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 1 and data['candidates']:
            id_ = data['candidates'][0]['id']
            accesskey = data['candidates'][0]['accesskey']
            lyric_url = f"https://lyrics.kugou.com/download?ver=1&client=pc&id={id_}&accesskey={accesskey}&fmt=lrc&charset=utf8"
            lyric_response = requests.get(lyric_url, headers=headers)
            if lyric_response.status_code == 200:
                lyric_data = lyric_response.json()
                if lyric_data['status'] == 1:
                    lyric = lyric_data['content']
                    # 去除时间戳
                    lyric = re.sub(r'\[.*?\]', '', lyric)
                    return lyric.strip()
    return None

# 清理文本函数
def clean_text(text):
    # 去除多余空格和空行
    if not text:
        return ''
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(non_empty_lines)
    # 去除多余空格，但保留换行符
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除特殊符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n，。❀！？：,!.?: ]', '', text)
    return text


def main():
    artists = ['ヨルシカ', 'n-buna']
    all_lyrics = []
    platforms = [
        ('QQ音乐', qqmusic_search, qqmusic_get_lyrics),
        ('网易云音乐', netease_search, netease_get_lyrics),
        ('酷狗音乐', kugou_search, kugou_get_lyrics)
    ]

    # 记录每个平台的下载状态
    platform_stats = {name: {'total': 0, 'success': 0} for name, _, _ in platforms}

    for artist in artists:
        print(f"\n开始下载歌手 {artist} 的歌词...")
        
        for platform_name, search_func, get_lyrics_func in platforms:
            print(f"\n尝试从 {platform_name} 下载...")
            try:
                song_ids = search_func(artist)
                if not song_ids:
                    print(f"- {platform_name} 未找到歌曲")
                    continue
                
                platform_stats[platform_name]['total'] += len(song_ids)
                
                for song_id in song_ids:
                    try:
                        lyrics = get_lyrics_func(song_id)
                        lyrics = clean_text(lyrics)
                        if lyrics:
                            all_lyrics.append(lyrics)
                            platform_stats[platform_name]['success'] += 1
                            print(f"- 成功下载歌词 (ID: {song_id})")
                        else:
                            print(f"- 歌词为空 (ID: {song_id})")
                    except Exception as e:
                        print(f"- 下载失败 (ID: {song_id}): {str(e)}")
                        
            except Exception as e:
                print(f"- {platform_name} 搜索失败: {str(e)}")

    # 打印统计信息
    print("\n下载统计:")
    for platform_name, stats in platform_stats.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{platform_name}:")
        print(f"- 总数: {stats['total']}")
        print(f"- 成功: {stats['success']}")
        print(f"- 成功率: {success_rate:.1f}%")

    # 将所有歌词合并为一个字符串
    combined_lyrics = '\n\n'.join(all_lyrics)
    print(f"\n总共下载成功歌词数: {len(all_lyrics)}")
    # 保存为文本文件
    with open('lyrics.txt', 'w', encoding='utf-8') as f:
        f.write(combined_lyrics)

if __name__ == "__main__":
    main()