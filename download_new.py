import os
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse
from tqdm.asyncio import tqdm
import time
import random

# ================= 配置区域 =================
LINKS_FILE = "movie.txt"
OUTPUT_DIR = r"F:\dataset\movie\moviebench"

# 1. 代理设置 (非常重要！如果你在中国，请务必填写)
# 格式示例: "http://127.0.0.1:7890" (查看你的VPN软件设置端口)
# 如果不需要代理，请设为 None
PROXY_URL = None  # e.g., "http://127.0.0.1:7890"

# 2. 并发数
# 有了断点续传和代理，可以适当提高到 5-10
CONCURRENCY = 1  # 减少并发数以避免服务器拒绝请求

# 账号密码
USERNAME1 = "G6Gz"
PASSWORD1 = "SookaiX1iu"
USERNAME2 = "chenlaneva@mails.cuc.edu.cn"
PASSWORD2 = "25B116A27F93D6036D46"

# 实时日志文件
FAILED_LOG = "failed_downloads_realtime.txt"
SUCCESS_LOG = "successful_downloads.txt"
# ===========================================

def get_random_credentials():
    """随机选择一组账号密码"""
    accounts = [
        (USERNAME1, PASSWORD1),
        (USERNAME2, PASSWORD2)
    ]
    return random.choice(accounts)

def fix_url_with_auth(url, user, pwd):
    url = url.strip()
    # 替换旧域名
    if "datasets.d2.mpi-inf.mpg.de/movieDescription" in url:
        url = url.replace(
            "http://datasets.d2.mpi-inf.mpg.de/movieDescription", 
            "https://moviedescription.mpi-inf.mpg.de"
        )
    
    # 注入账号密码
    if "://" in url and "@" not in url:
        protocol, rest = url.split("://", 1)
        return f"{protocol}://{user}:{pwd}@{rest}"
    return url

async def download_one_file(session, sem, raw_url, pbar, stats):
    """
    下载单个文件 (支持断点续传)
    """
    async with sem:  # 限制并发
        username, password = get_random_credentials()
        
        if not raw_url.strip():
            pbar.update(1)
            return raw_url, "skipped"
            
        auth_url = fix_url_with_auth(raw_url, username, password)
        
        # --- 解析路径和文件名 ---
        try:
            parsed = urlparse(auth_url)
            filename = os.path.basename(parsed.path)
            path_parts = parsed.path.split('/')
            movie_dir_name = path_parts[-2] if len(path_parts) >= 2 else "misc"
        except:
            pbar.update(1)
            return raw_url, "url_parse_error"

        save_dir = os.path.join(OUTPUT_DIR, movie_dir_name)
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # --- 智能跳过机制：检查文件是否已存在且大小合理 ---
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            # 如果文件存在且大于1KB，则认为已成功下载，跳过
            if file_size > 1024:
                stats['exist'] += 1
                pbar.set_postfix({"Exist": stats['exist'], "OK": stats['ok'], "Failed": stats['fail']})
                pbar.update(1)
                return raw_url, "exists_verified"

        # --- 断点续传逻辑 ---
        resume_header = {}
        file_mode = 'wb'
        downloaded_size = 0
        
        if os.path.exists(save_path):
            downloaded_size = os.path.getsize(save_path)
            # 如果文件已经存在且不为0，尝试续传
            if downloaded_size > 0:
                resume_header = {'Range': f'bytes={downloaded_size}-'}
                file_mode = 'ab' # 追加模式

        # 总体超时设置 (连接15秒，读取600秒)
        timeout = aiohttp.ClientTimeout(total=1200, sock_connect=15, sock_read=60)

        last_status_code = None  # 记录最后一次失败的状态码
        last_error_msg = ""      # 记录最后一次失败的错误信息

        for attempt in range(5): # 增加重试次数
            try:
                # 只有在非第一此尝试且文件存在时，才考虑是不是文件下坏了，需要重置
                # 这里简化处理：只要是 Range 请求 416 (范围错误)，说明文件可能变了或已完成
                
                async with session.get(auth_url, headers=resume_header, timeout=timeout, proxy=PROXY_URL) as resp:
                    
                    # 情况A: 文件已完全下载 (服务器返回 416 Range Not Satisfiable)
                    if resp.status == 416:
                        # 更新统计信息
                        stats['exist'] += 1
                        pbar.set_postfix({"Exist": stats['exist'], "OK": stats['ok'], "Failed": stats['fail']})
                        pbar.update(1)
                        return raw_url, "exists_verified"

                    # 情况B: 正常下载 (200) 或 续传 (206)
                    if resp.status in [200, 206]:
                        async with aiofiles.open(save_path, file_mode) as f:
                            async for chunk in resp.content.iter_chunked(64 * 1024): # 1MB chunk
                                await f.write(chunk)
                           
                        # 验证下载的文件是否有效（大小合理）
                        if os.path.getsize(save_path) > 0:
                            # 更新统计信息
                            stats['ok'] += 1
                            pbar.set_postfix({"Exist": stats['exist'], "OK": stats['ok'], "Failed": stats['fail']})
                            pbar.update(1)
                            return raw_url, "success"
                        else:
                            # 文件大小为0，可能是下载问题
                            last_status_code = "Empty File"
                            last_error_msg = "Downloaded file is empty"
                            await asyncio.sleep(random.uniform(2, 5))  # 稍微延迟再重试
                            continue
                    
                    # 情况C: 鉴权失败
                    if resp.status == 401:
                        # 换个账号重试
                        username, password = get_random_credentials()
                        auth_url = fix_url_with_auth(raw_url, username, password)
                        last_status_code = resp.status
                        last_error_msg = f"Unauthorized - {resp.status}"
                        continue 
                    
                    # 服务器错误，可能需要稍后重试
                    elif resp.status >= 500:
                        last_status_code = resp.status
                        last_error_msg = f"Server Error - {resp.status}"
                        # 指数退避策略
                        await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                    
                    # 客户端错误，如404等
                    elif resp.status >= 400:
                        last_status_code = resp.status
                        last_error_msg = f"Client Error - {resp.status}"
                        # 对于客户端错误，可能不需要重试，但仍然重试几次以处理临时错误
                        if attempt < 2:  # 只重试2次，然后标记为失败
                            await asyncio.sleep(random.uniform(1, 3))
                            continue
                        else:
                            break  # 超过重试次数，跳出循环
                        
                    # 其他错误，稍作等待后重试
                    else:
                        last_status_code = resp.status
                        last_error_msg = f"HTTP Error - {resp.status}"
                        await asyncio.sleep(random.uniform(1, 3))
                        continue

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # 网络错误，等待重试
                last_status_code = "Network Error"
                last_error_msg = f"Network Error - {str(e)}"
                # 指数退避策略
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
            except Exception as e:
                # 未知错误
                last_status_code = "Exception"
                last_error_msg = f"General Exception - {str(e)}"
                # 发生严重错误，直接跳出
                break

        # 如果重试都失败了
        stats['fail'] += 1
        pbar.set_postfix({"Exist": stats['exist'], "OK": stats['ok'], "Failed": stats['fail']})
        pbar.update(1)
        
        # 返回带有具体错误信息的状态
        if isinstance(last_status_code, int):
            return raw_url, f"fail_max_retries_status_{last_status_code}"
        else:
            return raw_url, f"fail_max_retries_{last_error_msg}"

async def main():
    if not os.path.exists(LINKS_FILE):
        print(f"❌ 找不到文件: {LINKS_FILE}")
        return

    with open(LINKS_FILE, 'r') as f:
        links = [line.strip() for line in f if line.strip().startswith("http")]
    
    print(f"Total Tasks: {len(links)}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Proxy      : {PROXY_URL if PROXY_URL else 'Disabled (建议开启代理以提速)'}")
    print("-" * 40)

    # TCP 连接池设置
    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 5, ttl_dns_cache=300, ssl=False)
    
    # 限制并发的信号量
    sem = asyncio.Semaphore(CONCURRENCY)
    
    # 初始化统计字典
    stats = {"ok": 0, "exist": 0, "fail": 0}
    
    # 进度条，显示实时统计
    pbar = tqdm(total=len(links), unit="file", ncols=100, postfix={"Exist": 0, "OK": 0, "Failed": 0})

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        # --- 关键修改：一次性创建所有任务，而不是在循环里 await ---
        for url in links:
            task = asyncio.create_task(download_one_file(session, sem, url, pbar, stats))
            tasks.append(task)
        
        # 并发执行并等待所有结果
        results = await asyncio.gather(*tasks)

    # 写入日志
    pbar.close()
    
    with open(FAILED_LOG, "w", encoding='utf-8') as f:
        f.write(f"--- Download Report: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for url, status in results:
            if status == "exists_verified":
                stats["ok"] += 1  # 在这里重新统计一次，因为exists_verified算作成功
            elif status == "skipped":
                pass
            elif "fail_max_retries" in status:
                # 将失败的URL写入日志，包含具体错误信息
                f.write(f"{url} -> {status}\n")
            else:
                # 其他错误状态也写入日志
                if status != "success" and status != "url_parse_error":
                    f.write(f"{url} -> {status}\n")
    
    # 同时记录成功下载的文件
    with open(SUCCESS_LOG, "w", encoding='utf-8') as f:
        f.write(f"--- Successful Downloads: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for url, status in results:
            if status == "success" or status == "exists_verified":
                f.write(f"{url} -> {status}\n")

    print(f"\n✅ 全部完成!")
    print(f"存在: {stats['exist']}, 成功下载: {stats['ok']}, 失败: {stats['fail']}")
    print(f"失败日志: {FAILED_LOG}")
    print(f"成功日志: {SUCCESS_LOG}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())