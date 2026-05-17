#!/bin/bash

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` <filesToDownload> <username> <password> [parallelDownloads=2]"
  exit $E_BADARGS
fi

filesToDownload=$1
username=$2
password=$3
parallelDownloads=$4

if [ $# -lt 4 ]
then
	parallelDownloads=2
fi

# 函数：下载单个文件，如果目标已存在则跳过
download_file() {
    local url=$1
    
    # 解析URL以获取目录和文件名
    local base_url=$(dirname "$url")
    local dir=$(basename "$(dirname "$base_url")")
    local filename=$(basename "$url")
    
    # 目标目录
    local target_dir="/f/dataset/movie/moviebench/$dir"
    mkdir -p "$target_dir"
    
    # 检查文件是否已存在且大小大于1KB
    if [ -f "$target_dir/$filename" ]; then
        current_size=$(stat -f%z "$target_dir/$filename" 2>/dev/null || stat -c%s "$target_dir/$filename" 2>/dev/null)
        if [ "$current_size" -gt 1024 ]; then
            echo "File $target_dir/$filename already exists, skipping..."
            return 0
        fi
    fi
    
    # 尝试下载文件
    wget -nc -rnH --cut-dirs=2 -q --user="$username" --password="$password" \
         --directory-prefix="$target_dir" "$url"
    
    # 检查下载是否成功
    if [ $? -ne 0 ]; then
        echo "Failed to download $url"
        return 1
    fi
}

export -f download_file
export username password

# 读取文件列表并并行下载
cat "$filesToDownload" | xargs -n 1 -P "$parallelDownloads" -I {} bash -c 'download_file "$@"' _ {}