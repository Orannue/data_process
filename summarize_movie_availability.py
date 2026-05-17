import os
from urllib.parse import urlparse


def summarize_movie_availability():
    """
    统计movie.txt中每个链接对应的.avi文件的存在情况
    """
    movie_txt_path = r"F:\dataset\movie\TransNetV2\movie.txt"
    base_output_dir = r"F:\dataset\movie\moviebench"
    
    if not os.path.exists(movie_txt_path):
        print(f"❌ 找不到文件: {movie_txt_path}")
        return

    with open(movie_txt_path, 'r') as f:
        links = [line.strip() for line in f if line.strip().startswith("http")]

    total_links = len(links)
    print(f"总共 {total_links} 个链接需要检查")
    
    found_count = 0
    missing_count = 0
    missing_links = []  # 记录缺失的链接
    
    for i, link in enumerate(links, 1):
        # 解析URL获取文件名和目录名
        try:
            parsed = urlparse(link)
            filename = os.path.basename(parsed.path)
            path_parts = parsed.path.split('/')
            movie_dir_name = path_parts[-2] if len(path_parts) >= 2 else "misc"
        except:
            print(f"[{i}/{total_links}] 无法解析URL: {link}")
            missing_links.append((link, "URL解析错误"))
            missing_count += 1
            continue

        # 构建保存路径
        save_dir = os.path.join(base_output_dir, movie_dir_name)
        save_path = os.path.join(save_dir, filename)
        
        # 检查文件是否存在
        if os.path.exists(save_path):
            found_count += 1
        else:
            missing_links.append((link, "文件不存在"))
            missing_count += 1
            
        # 每10000个链接打印一次进度
        if i % 10000 == 0:
            print(f"[{i}/{total_links}] 已检查 {i} 个链接，目前找到 {found_count} 个，缺失 {missing_count} 个")

    print(f"\n检查完成!")
    print(f"存在的文件: {found_count}")
    print(f"缺失的文件: {missing_count}")
    print(f"总体进度: {found_count}/{total_links} ({found_count/total_links*100:.2f}%)")
    
    # 保存缺失的链接到文件
    if missing_links:
        missing_file_path = "missing_movies.txt"
        with open(missing_file_path, 'w') as f:
            for link, reason in missing_links:
                f.write(f"{link}\n")
        print(f"缺失的 {len(missing_links)} 个链接已保存到 {missing_file_path}")
    
    return missing_links


if __name__ == "__main__":
    summarize_movie_availability()