from opencc import OpenCC

def traditional_to_simplified(file_path):
    # 创建一个 OpenCC 对象，指定转换模式为 't2s'（繁体转简体）
    cc = OpenCC('t2s')
    try:
        # 以只读模式打开歌词文件，并使用 UTF-8 编码
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件中的所有内容
            lyrics = file.read()
        # 使用 OpenCC 对象将繁体字转换为简体字
        simplified_lyrics = cc.convert(lyrics)
        # 以写入模式打开文件，将转换后的简体字歌词写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(simplified_lyrics)
        print("歌词中的繁体字已成功转换为简体字。")
    except FileNotFoundError:
        print(f"未找到指定的文件: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 替换为你的歌词文件的实际路径
    file_path = 'my_transformer/data/lyrics.txt'
    traditional_to_simplified(file_path)