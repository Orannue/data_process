import json
import os
import numpy as np
import cv2
from pathlib import Path


def is_video_valid(video_path):
    """
    Check if video file is valid and not corrupted.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to read a few frames to ensure the video is not corrupted
        valid_frames = 0
        for _ in range(10):  # Check first 10 frames
            ret, _ = cap.read()
            if ret:
                valid_frames += 1
            else:
                break
        
        cap.release()
        
        # Consider video valid if at least some frames can be read
        return valid_frames > 0
    except:
        return False


def parse_timestamp_key(filename):
    """Extract timestamp from filename for sorting"""
    try:
        # Extract timestamp part from full filename (the part after the last underscore)
        parts = filename.split('_')
        time_part = parts[-1]
        start_time, end_time = time_part.split('-')
        
        def time_to_seconds(time_str):
            h, m, s = time_str.split('.')
            return int(h)*3600 + int(m)*60 + float(f"{s[:2]}.{s[2:]}")
        
        start_seconds = time_to_seconds(start_time)
        return start_seconds
    except:
        # If parsing fails, return 0 to sort at the beginning
        return 0


def get_original_frames_for_save(video_path):
    """
    Extract all frames from a video file.
    """
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
            frames.append(frame)  # Keep BGR format for OpenCV
        
        cap.release()
        return frames
    
    except Exception as e:
        print(f"    Error extracting original frames from {video_path}: {str(e)}")
        return []


def save_merged_videos(json_data, moviebench_path, output_base_path):
    """
    Process all movies in the JSON data, merge all videos in each scene and save as total.mp4
    """
    # Create base output directory if it doesn't exist
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    for movie_id, scenes in json_data.items():
        print(f"\nProcessing movie: {movie_id}")
        
        # Get the path for this movie
        movie_path = os.path.join(moviebench_path, movie_id)
        
        if not os.path.exists(movie_path):
            print(f"Movie path does not exist: {movie_path}")
            continue

        # Process each scene in the movie
        for scene_idx, (scene_desc, scene_videos) in enumerate(scenes.items()):
            print(f"  Processing scene: {scene_desc}")
            
            # Collect all video files for this scene with their timestamps
            video_info_list = []
            all_exist_and_valid = True
            
            # Sort the scene_videos by timestamp to ensure temporal order
            sorted_scene_videos = sorted(scene_videos, key=parse_timestamp_key)
            
            for video_name in sorted_scene_videos:
                video_full_path = os.path.join(movie_path, f"{video_name}.avi")
                
                # Check if file exists and is valid
                if not os.path.exists(video_full_path):
                    print(f"    Warning: Video file does not exist: {video_full_path}")
                    all_exist_and_valid = False
                    break
                elif not is_video_valid(video_full_path):
                    print(f"    Warning: Video file is corrupted: {video_full_path}")
                    all_exist_and_valid = False
                    break
                else:
                    video_info_list.append({
                        'path': video_full_path,
                        'name': video_name
                    })
            
            if not all_exist_and_valid or not video_info_list:
                print(f"    Skipping scene due to missing or corrupted video files: {scene_desc}")
                continue
            
            # Create a more descriptive folder name including movie and scene info
            scene_name_clean = "".join(c for c in scene_desc.replace("Sence", "Scene") if c.isalnum() or c in (' ', '-', '_')).rstrip()
            scene_name_clean = scene_name_clean.replace(" ", "_").replace("__", "_")
            scene_output_path = os.path.join(output_base_path, movie_id, scene_name_clean)
            
            # Now create the directory since we know all conditions are met
            Path(scene_output_path).mkdir(parents=True, exist_ok=True)
            
            # Concatenate all videos in the scene
            original_video_segments = []  # Store original quality segments for final output
            
            for video_info in video_info_list:
                video_path = video_info['path']
                print(f"    Processing video: {os.path.basename(video_path)}")
                
                # Get original quality frames (full-res for output)
                original_frames = get_original_frames_for_save(video_path)
                original_video_segments.extend(original_frames)
                
                if len(original_frames) == 0:
                    print(f"      Warning: Could not extract frames from {video_path}")
            
            if not original_video_segments:
                print(f"    No frames extracted for scene: {scene_desc}")
                # Remove the directory if no frames were extracted
                if os.path.exists(scene_output_path):
                    os.rmdir(scene_output_path)
                continue

            print(f"    Total frames for scene: {len(original_video_segments)}")
            
            # Create the total.mp4 video file
            if len(original_video_segments) > 0:
                total_video_path = os.path.join(scene_output_path, "total.mp4")
                
                # Get frame dimensions from the first frame
                height, width, layers = original_video_segments[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for better quality
                
                # Get FPS from the first video in the scene
                cap_first = cv2.VideoCapture(video_info_list[0]['path'])
                fps = cap_first.get(cv2.CAP_PROP_FPS)
                cap_first.release()
                actual_fps = fps if fps > 0 else 25
                
                # Create video writer
                out = cv2.VideoWriter(total_video_path, fourcc, 16, (width, height))
                
                # Write all frames to the video
                for frame in original_video_segments:
                    out.write(frame)  # Original frames are already in BGR format
                
                out.release()
                
                # Verify the written file
                if os.path.exists(total_video_path):
                    cap_test = cv2.VideoCapture(total_video_path)
                    if cap_test.isOpened():
                        total_frame_count = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap_test.release()
                        
                        if total_frame_count > 0:
                            print(f"      Saved merged video to {total_video_path} ({total_frame_count} frames)")
                        else:
                            print(f"      Error: Generated video file appears to be corrupted: {total_video_path}")
                    else:
                        print(f"      Error: Could not open generated video file: {total_video_path}")
                else:
                    print(f"      Error: Could not create video file: {total_video_path}")


def main():
    # Load the JSON file containing scene information
    json_path = r"F:\dataset\movie\movies_scenes.json"
    moviebench_path = r"F:\dataset\movie\moviebench"
    output_base_path = r"F:\dataset\movie\shots_output2"
    
    print("Loading scene data...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    print("Starting processing...")
    save_merged_videos(json_data, moviebench_path, output_base_path)
    print("Processing complete!")


if __name__ == "__main__":
    main()