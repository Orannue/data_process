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

# 函数：下载单个文件，带重试机制
download_file() {
    local url=$1
    local max_attempts=3
    
    # 解析URL以获取目录和文件名
    local path_part=$(echo "$url" | sed -e 's/.*\/\([^\/]*\/[^\/]*\)$/\1/')
    local dir=$(echo "$path_part" | cut -d'/' -f1)
    local filename=$(echo "$path_part" | cut -d'/' -f2)
    
    # 如果无法正确解析，使用备用方法
    if [ "$dir" = "$path_part" ] || [ -z "$dir" ]; then
        dir=$(basename "$(dirname "$url")")
        filename=$(basename "$url")
    fi
    
    # 目标目录
    local target_dir="./downloads/$dir"
    mkdir -p "$target_dir"
    
    local target_path="$target_dir/$filename"
    
    # 检查文件是否已存在且大小大于1KB
    if [ -f "$target_path" ]; then
        current_size=$(stat -f%z "$target_path" 2>/dev/null || stat -c%s "$target_path" 2>/dev/null)
        if [ "$current_size" -gt 1024 ]; then
            echo "File $target_path already exists, skipping..."
            return 0
        fi
    fi
    
    # 尝试下载，带重试机制
    for attempt in $(seq 1 $max_attempts); do
        if [ $attempt -gt 1 ]; then
            echo "Retrying ($attempt/$max_attempts) for $url"
            sleep $((attempt * 2))  # 指数退避
        fi
        
        # 使用更完整的wget参数，包括重定向处理
        wget -t 3 -T 60 -O "$target_path" -q --user="$username" --password="$password" "$url"
        
        # 检查下载是否成功
        if [ $? -eq 0 ] && [ -s "$target_path" ]; then
            echo "Successfully downloaded $filename"
            return 0
        else
            # 删除可能损坏的文件
            rm -f "$target_path" 2>/dev/null
        fi
    done
    
    echo "Failed to download $url after $max_attempts attempts"
    return 1
}

export -f download_file
export username password

# 读取文件列表并并行下载
while IFS= read -r url; do
    if [[ $url =~ ^https?:// ]]; then
        # 在后台启动下载进程，控制并发数量
        download_file "$url" &
        
        # 控制并发数量
        if (((i=i%parallelDownloads)<parallelDownloads)); then ((i++==parallelDownloads-1)) && wait
    fi
done < "$filesToDownload"

# 等待所有后台进程完成
wait