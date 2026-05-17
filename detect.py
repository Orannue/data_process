import json
import os
import re
import cv2
import numpy as np
from pathlib import Path

# 引入 PySceneDetect
from scenedetect import open_video, SceneManager, ContentDetector

def detect_scenes_via_temp_file(frames, output_dir, fps=25.0, threshold=27.0):
    """
    1. 将内存帧写入临时 MP4 文件 (使用 mp4v，兼容性最好)
    2. 使用 PySceneDetect 检测
    3. 删除临时文件
    4. 返回切分点列表
    """
    if not frames:
        return []

    # --- 修正点 1: 文件名后缀改回 .mp4 ---
    temp_filename = os.path.join(output_dir, "temp_processing_video.mp4")
    
    h, w = frames[0].shape[:2]
    
    # --- 修正点 2: 编码改回 mp4v，解决 'MJPG' is not supported 的报错 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    
    scenes_frames = []
    video = None
    
    try:
        # 2. 检测
        video = open_video(temp_filename)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # 这里的 downscale_factor 可以提高检测速度，默认为 1 (不缩放)
        # 如果觉得慢可以设为 2，但为了精准度建议保持默认
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        
        # 3. 转换结果
        if not scene_list:
            scenes_frames.append([0, len(frames) - 1])
        else:
            for scene in scene_list:
                start, end = scene
                # PySceneDetect 的 end 是开区间，需转换为闭区间索引
                scenes_frames.append([start.get_frames(), end.get_frames() - 1])
                
    except Exception as e:
        print(f"    Error during detection: {e}")
        scenes_frames.append([0, len(frames) - 1])
        
    finally:
        # 4. 清理资源 (包含文件锁释放逻辑)
        if video is not None:
            if hasattr(video, 'release'):
                video.release()
            del video 
            
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                # 依然删不掉也没关系，不影响程序运行，只是多占点空间
                print(f"    Warning: Could not delete temp file: {e}")
                
    return scenes_frames


def is_video_valid(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        valid_frames = 0
        for _ in range(10):  # Check first 10 frames
            ret, _ = cap.read()
            if ret:
                valid_frames += 1
            else:
                break
        cap.release()
        return valid_frames > 0
    except:
        return False


def parse_timestamp_key(filename):
    try:
        parts = filename.split('_')
        time_part = parts[-1]
        start_time, end_time = time_part.split('-')
        
        def time_to_seconds(time_str):
            h, m, s = time_str.split('.')
            return int(h)*3600 + int(m)*60 + float(f"{s[:2]}.{s[2:]}")
        
        start_seconds = time_to_seconds(start_time)
        return start_seconds
    except:
        return 0


SINGLE_PERSON_POSITIVE_PATTERN = re.compile(
    r"\b(close-up|close up|face|portrait|bedroom|hospital room|office|study|library|"
    r"room|window|rowboat|interior with|inside a house|cozy dining room)\b",
    re.IGNORECASE,
)

SINGLE_PERSON_NEGATIVE_PATTERN = re.compile(
    r"\b(crowd|battle|battlefield|soldiers|party|ballroom|hall|train station|street|"
    r"market|audience|church|banquet|wedding|people|group)\b",
    re.IGNORECASE,
)


def score_scene_priority(scene_desc, scene_videos):
    """
    Heuristic score: scenes with fewer source clips and more single-character-like
    descriptions are processed first to speed up downstream shot/person discovery.
    """
    score = 0.0
    clip_count = len(scene_videos)

    if clip_count <= 2:
        score += 3.0
    elif clip_count <= 4:
        score += 1.5
    elif clip_count >= 10:
        score -= 1.5

    if SINGLE_PERSON_POSITIVE_PATTERN.search(scene_desc):
        score += 2.0
    if SINGLE_PERSON_NEGATIVE_PATTERN.search(scene_desc):
        score -= 2.5

    return score


def score_movie_priority(scenes):
    if not scenes:
        return float("-inf")
    total_score = sum(
        score_scene_priority(scene_desc, scene_videos)
        for scene_desc, scene_videos in scenes.items()
    )
    return total_score / len(scenes)


def get_original_frames_for_save(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Could not open video file: {video_path}")
            return []
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"    Error extracting original frames from {video_path}: {str(e)}")
        return []


def process_movie_scenes(json_data, moviebench_path, output_base_path):
    # Create base output directory if it doesn't exist
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    # sorted_movies = sorted(
    #     json_data.items(),
    #     key=lambda item: score_movie_priority(item[1]),
    #     reverse=True,
    # )
    sorted_movies = json_data.items()
    #print(f"Total movies to process: {len(sorted_movies)}")

    for movie_rank, (movie_id, scenes) in enumerate(sorted_movies, start=1):
        movie_priority = score_movie_priority(scenes)
        print(f"\nProcessing movie: {movie_id}")
        print(f"  Movie priority rank: {movie_rank}/{len(sorted_movies)} | score={movie_priority:.2f}")
        movie_path = os.path.join(moviebench_path, movie_id)
        
        if not os.path.exists(movie_path):
            print(f"Movie path does not exist: {movie_path}")
            continue

        sorted_scenes = sorted(
            scenes.items(),
            key=lambda item: score_scene_priority(item[0], item[1]),
            reverse=True,
        )

        for scene_idx, (scene_desc, scene_videos) in enumerate(sorted_scenes):
            scene_priority = score_scene_priority(scene_desc, scene_videos)
            print(f"  Processing scene: {scene_desc}")
            print(f"    Scene priority score: {scene_priority:.2f}")
            
            video_info_list = []
            all_exist_and_valid = True
            sorted_scene_videos = sorted(scene_videos, key=parse_timestamp_key)
            
            for video_name in sorted_scene_videos:
                video_full_path = os.path.join(movie_path, f"{video_name}.avi")
                if not os.path.exists(video_full_path):
                    print(f"    Warning: Video file does not exist: {video_full_path}")
                    all_exist_and_valid = False
                    break
                elif not is_video_valid(video_full_path):
                    print(f"    Warning: Video file is corrupted: {video_full_path}")
                    all_exist_and_valid = False
                    break
                else:
                    video_info_list.append({'path': video_full_path, 'name': video_name})
            
            if not all_exist_and_valid or not video_info_list:
                print(f"    Skipping scene due to missing or corrupted video files: {scene_desc}")
                continue
            
            # Create output directory
            scene_name_clean = "".join(c for c in scene_desc.replace("Sence", "Scene") if c.isalnum() or c in (' ', '-', '_')).rstrip()
            scene_name_clean = scene_name_clean.replace(" ", "_").replace("__", "_")
            scene_output_path = os.path.join(output_base_path, movie_id, scene_name_clean)
            
            if os.path.exists(scene_output_path) and any(Path(scene_output_path).iterdir()):
                print(f"    Skipping already processed scene: {scene_desc}")
                continue
            
            Path(scene_output_path).mkdir(parents=True, exist_ok=True)
            
            # --- 步骤 1: 将所有视频帧读取并拼接 (内存中) ---
            original_video_segments = []
            fps_for_detection = 25.0
            
            # 记录第一个视频的原始帧率，后续检测和导出都沿用它
            if len(video_info_list) > 0:
                cap_fps = cv2.VideoCapture(video_info_list[0]['path'])
                fps_val = cap_fps.get(cv2.CAP_PROP_FPS)
                if fps_val > 0:
                    fps_for_detection = fps_val
                cap_fps.release()

            for video_info in video_info_list:
                video_path = video_info['path']
                frames = get_original_frames_for_save(video_path)
                if len(frames) == 0:
                    print(f"      Warning: Could not extract frames from {video_path}")
                    continue
                original_video_segments.extend(frames)
            
            if not original_video_segments:
                print(f"    No frames extracted for scene: {scene_desc}")
                Path(scene_output_path).rmdir()
                continue

            print(f"    Total stitched frames: {len(original_video_segments)}")
            print(f"    Source FPS: {fps_for_detection:.3f}")
            
            # --- 步骤 2: 使用临时文件进行 PySceneDetect 检测 ---
            shot_segments = detect_scenes_via_temp_file(
                original_video_segments, 
                scene_output_path,  # 临时文件存在这个文件夹里
                fps=fps_for_detection, 
                threshold=30.0
            )
            
            print(f"    Detected {len(shot_segments)} shots.")
            
            # --- 步骤 3: 保存切分好的 Shot (保持原逻辑) ---
            for shot_idx, (start_frame, end_frame) in enumerate(shot_segments):
                # 利用索引从内存的大列表中直接切片，不损失画质
                shot_frames = original_video_segments[start_frame:end_frame+1]
                
                if len(shot_frames) > 0:
                    start_seconds = start_frame / fps_for_detection
                    end_seconds = end_frame / fps_for_detection
                    
                    start_h = int(start_seconds // 3600)
                    start_m = int((start_seconds % 3600) // 60)
                    start_s = start_seconds % 60
                    start_ts = f"{start_h:02d}.{start_m:02d}.{start_s:06.3f}"
                    
                    end_h = int(end_seconds // 3600)
                    end_m = int((end_seconds % 3600) // 60)
                    end_s = end_seconds % 60
                    end_ts = f"{end_h:02d}.{end_m:02d}.{end_s:06.3f}"
                    
                    shot_filename = os.path.join(scene_output_path, f"{movie_id}_{scene_name_clean}_shot_{shot_idx+1}_{start_ts}-{end_ts}.mp4")
                    
                    height, width, layers = shot_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    
                    out = cv2.VideoWriter(shot_filename, fourcc, fps_for_detection, (width, height))
                    for frame in shot_frames:
                        out.write(frame) 
                    out.release()
                    
                    if os.path.exists(shot_filename) and os.path.getsize(shot_filename) > 0:
                        pass
                    else:
                        print(f"      Error: Generated video file is invalid: {shot_filename}")
                else:
                    print(f"      No frames to save for shot {shot_idx+1}")


def main():
    # 路径配置
    json_path = r"F:\dataset\movie\movies_scenes.json"
    moviebench_path = r"F:\dataset\movie\moviebench"
    output_base_path = r"H:\dataset\movie_shot"
    
    print("Loading scene data...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    print("Starting processing with PySceneDetect (Temp File Strategy)...")
    process_movie_scenes(json_data, moviebench_path, output_base_path)
    print("Processing complete!")


if __name__ == "__main__":
    main()