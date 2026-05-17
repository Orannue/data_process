import os
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import sys


def scan_directory(directory_path):
    """
    扫描指定目录，返回所有文件列表
    """
    directory_path = Path(directory_path)
    all_files = []
    
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = Path(root) / file
            all_files.append(file_path)
        # 更新进度
        if len(all_files) % 100 == 0:  # 每添加100个文件更新一次进度
            print(f"\r已扫描 {len(all_files)} 个文件...", end='', flush=True)
    
    print(f"\n总共扫描到 {len(all_files)} 个文件")
    return all_files


def find_archive_files(files):
    """
    从文件列表中找出压缩包文件
    """
    archive_extensions = ['.zip', '.tar', '.tar.gz', '.tgz', '.rar']
    archives = []
    
    for file in files:
        if any(str(file).lower().endswith(ext) for ext in archive_extensions):
            archives.append(file)
    
    return archives


def get_files_from_archive(archive_path):
    """
    从压缩包中获取文件列表
    """
    archive_path = str(archive_path)
    print(f"正在读取压缩包: {archive_path}")
    
    files_in_archive = set()
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            # 获取文件总数
            total_files = len(zipf.namelist())
            print(f"压缩包包含 {total_files} 个文件")
            
            # 使用tqdm显示进度
            for i, file_name in enumerate(tqdm(zipf.namelist(), desc=f"读取 {Path(archive_path).name}", unit="file")):
                files_in_archive.add(file_name)
                
    elif archive_path.endswith('.tar'):
        try:
            with tarfile.TarFile(archive_path, 'r') as tarf:
                members = tarf.getmembers()
                total_members = len(members)
                print(f"压缩包包含 {total_members} 个文件")
                
                for member in tqdm(members, desc=f"读取 {Path(archive_path).name}", unit="file"):
                    files_in_archive.add(member.name)
        except tarfile.ReadError as e:
            print(f"无法读取tar文件 {archive_path}: {e}")
            return set()
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        try:
            with tarfile.open(archive_path, 'r:gz') as tarf:
                members = tarf.getmembers()
                total_members = len(members)
                print(f"压缩包包含 {total_members} 个文件")
                
                for member in tqdm(members, desc=f"读取 {Path(archive_path).name}", unit="file"):
                    files_in_archive.add(member.name)
        except tarfile.ReadError as e:
            print(f"无法读取tar.gz文件 {archive_path}: {e}")
            return set()
    elif archive_path.endswith('.rar'):
        # 如果有安装rarfile库，可以处理rar文件
        try:
            import rarfile
            with rarfile.RarFile(archive_path, 'r') as rar:
                file_list = rar.namelist()
                total_files = len(file_list)
                print(f"压缩包包含 {total_files} 个文件")
                
                for file_name in tqdm(file_list, desc=f"读取 {Path(archive_path).name}", unit="file"):
                    files_in_archive.add(file_name)
        except ImportError:
            print("警告: 未安装rarfile库，无法处理RAR文件")
            return set()
        except Exception as e:
            print(f"无法读取rar文件 {archive_path}: {e}")
            return set()
    
    print(f"压缩包 {archive_path} 包含 {len(files_in_archive)} 个文件")
    return files_in_archive


def get_all_compressed_files(archive_files, base_path):
    """
    获取所有压缩包中的文件集合
    """
    all_compressed_files = set()
    
    for i, archive in enumerate(archive_files):
        print(f"[{i+1}/{len(archive_files)}] 正在处理压缩包: {archive.name}")
        files_in_archive = get_files_from_archive(archive)
        
        # 将压缩包中的相对路径转换为绝对路径进行比较
        for file_in_archive in files_in_archive:
            # 处理路径中的特殊情况，如以'/'开头的路径
            if file_in_archive.startswith('/'):
                file_in_archive = file_in_archive[1:]
            
            full_path = (base_path / file_in_archive).resolve()
            all_compressed_files.add(full_path)
    
    return all_compressed_files


def pack_remaining_files(remaining_files, output_path, archive_format='zip'):
    """
    将剩余的文件打包到指定的压缩文件中
    """
    output_path = Path(output_path)
    
    if archive_format == 'zip':
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            base_path = Path(r"F:\dataset\movie\shots_output")
            # 显示打包进度
            pbar = tqdm(remaining_files, desc="打包文件", unit="file")
            for i, file in enumerate(pbar):
                # 保留相对路径结构
                arcname = file.relative_to(base_path).as_posix()
                zipf.write(file, arcname)
                # 每处理100个文件更新一次描述，这样用户可以看到进度
                if (i + 1) % 100 == 0:
                    pbar.set_postfix({"已处理": f"{i+1}/{len(remaining_files)}"})
    elif archive_format in ['tar', 'tar.gz']:
        mode = 'w:gz' if archive_format == 'tar.gz' else 'w'
        with tarfile.open(output_path, mode) as tarf:
            base_path = Path(r"F:\dataset\movie\shots_output")
            # 显示打包进度
            pbar = tqdm(remaining_files, desc="打包文件", unit="file")
            for i, file in enumerate(pbar):
                # 保留相对路径结构
                arcname = file.relative_to(base_path).as_posix()
                tarf.add(file, arcname=arcname)
                # 每处理100个文件更新一次描述
                if (i + 1) % 100 == 0:
                    pbar.set_postfix({"已处理": f"{i+1}/{len(remaining_files)}"})


def main():
    directory_path = r"F:\dataset\movie\shots_output"
    base_path = Path(directory_path)
    
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return
    
    print("开始扫描目录...")
    # 获取所有文件
    all_files = scan_directory(directory_path)
    
    print("查找压缩包...")
    # 找出所有的压缩包
    archive_files = find_archive_files(all_files)
    
    print(f"找到的压缩包有: {[str(f) for f in archive_files]}")
    
    if len(archive_files) < 2:
        print("在目录中找到的压缩包少于2个")
        return
    
    print("获取压缩包中的文件...")
    # 获取压缩包中的所有文件
    compressed_files = get_all_compressed_files(archive_files[:2], base_path)  # 只处理前两个压缩包
    print(f"压缩包中总共包含 {len(compressed_files)} 个文件")
    
    # 确定要排除的两个压缩包
    excluded_archives = archive_files[:2]  # 取前两个作为要排除的压缩包
    print(f"将排除以下压缩包: {[str(f) for f in excluded_archives]}")
    
    # 获取需要打包的文件：所有文件 - 压缩包文件 - 压缩包内的文件
    excluded_paths = set(excluded_archives)  # 排除压缩包本身
    remaining_files = [
        f for f in all_files 
        if f not in excluded_paths and f not in compressed_files
    ]
    
    print(f"需要打包的文件数量: {len(remaining_files)}")
    
    # 输出新的压缩包名称
    output_filename = os.path.join(
        os.path.dirname(directory_path), 
        "packed_remaining_files.zip"
    )
    
    print(f"开始打包文件到: {output_filename}")
    # 打包剩余文件
    pack_remaining_files(remaining_files, output_filename)
    
    print(f"打包完成! 输出文件: {output_filename}")


if __name__ == "__main__":
    main()