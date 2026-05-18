import json
import os
import torch
import numpy as np
import cv2
from inference_pytorch.transnetv2_pytorch import TransNetV2
from pathlib import Path


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    """
    Convert predictions to scene (shot) segments based on threshold.
    Returns array of [start_frame, end_frame] pairs representing shots.
    """
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:  # transition from scene boundary to normal frame
            start = i
        if t_prev == 0 and t == 1 and i != 0:  # transition from normal frame to scene boundary
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # fix case when all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


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


def predict_frames_torch(model, frames: np.ndarray):
    """
    Process frames similar to TensorFlow version's predict_frames function.
    Handles long videos by splitting into overlapping windows.
    """
    assert len(frames.shape) == 4 and frames.shape[1:] == (27, 48, 3), \
        "[TransNetV2-PyTorch] Input shape must be [frames, height, width, 3]."

    def input_iterator():
        # return windows of size 100 where the first/last 25 frames are from the previous/next batch
        # the first and last window must be padded by copies of the first and last frame of the video
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    predictions = []

    device = next(model.parameters()).device  # Get the device the model is on

    for inp in input_iterator():
        # Convert to tensor
        input_tensor = torch.tensor(inp, dtype=torch.uint8)
        
        with torch.no_grad():
            # Run prediction
            single_frame_pred, all_frame_pred = model(input_tensor.to(device))
            
            # Apply sigmoid to get probabilities
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
        
        # Extract middle 50 frames (removing padded frames)
        predictions.append((single_frame_pred[0, 25:75, 0],
                            all_frame_pred[0, 25:75, 0]))

        print("\r[TransNetV2-PyTorch] Processing video frames {}/{}".format(
            min(len(predictions) * 50, len(frames)), len(frames)
        ), end="")

    print("")  # New line after progress indicator

    # Concatenate predictions removing the padding
    single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
    all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

    # Return only the original frames (remove any extra padded frames)
    return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]


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


def process_movie_scenes(json_data, moviebench_path, output_base_path, model):
    """
    Process all movies in the JSON data, split scenes into shots using TransNetV2.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

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
            
            # Check if this scene has already been processed by checking if output directory is not empty
            if os.path.exists(scene_output_path) and any(Path(scene_output_path).iterdir()):
                print(f"    Skipping already processed scene: {scene_desc}")
                continue
            
            # Now create the directory since we know all conditions are met
            Path(scene_output_path).mkdir(parents=True, exist_ok=True)
            
            # Concatenate all videos in the scene and process - this is the key change
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
                Path(scene_output_path).rmdir()
                continue

            print(f"    Total frames for scene: {len(original_video_segments)}")
            
            # Convert frames to the required format for the model
            # Resize frames to model input size (27, 48, 3)
            resized_frames = []
            for frame in original_video_segments:
                # Resize frame to required dimensions for model processing
                resized_frame = cv2.resize(frame, (48, 27))  # Note: cv2.resize takes (width, height)
                # Convert from BGR (OpenCV default) to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                resized_frames.append(rgb_frame)
            
            all_frames = np.array(resized_frames, dtype=np.uint8)
            print(f"    Resized frames shape: {all_frames.shape}")
            
            # Process with TransNetV2 using the predict_frames approach
            single_frame_pred, all_frame_pred = predict_frames_torch(model, all_frames)
            
            # Convert predictions to scenes/shots using our own function
            shot_segments = predictions_to_scenes(all_frame_pred)
                    
            print(f"    Detected {len(shot_segments)} shots in scene")
            
            # Calculate cumulative frame counts for each sub-video to map shots back correctly
            sub_video_frame_counts = []
            cumulative_frame_count = 0
            cumulative_frame_counts = [0]  # Start with 0 for the first video
            
            for video_info in video_info_list:
                video_path = video_info['path']
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                sub_video_frame_counts.append(frame_count)
                cumulative_frame_count += frame_count
                cumulative_frame_counts.append(cumulative_frame_count)
            
            # Save each shot as a separate video file with unique names
            for shot_idx, (start_frame, end_frame) in enumerate(shot_segments):
                # Determine which original video files this shot spans
                # Find the original frames for this shot
                shot_frames = original_video_segments[start_frame:end_frame+1]
                
                if len(shot_frames) > 0:
                    # Create unique shot name based on original video segments and shot timing
                    # Calculate approximate timestamp based on frame numbers
                    fps_source = cv2.VideoCapture(video_info_list[0]['path'])
                    fps = fps_source.get(cv2.CAP_PROP_FPS)
                    fps_source.release()
                    actual_fps = fps if fps > 0 else 25
                    
                    start_seconds = start_frame / actual_fps
                    end_seconds = end_frame / actual_fps
                    
                    # Format as HH.MM.SSS-HH.MM.SSS
                    start_h = int(start_seconds // 3600)
                    start_m = int((start_seconds % 3600) // 60)
                    start_s = start_seconds % 60
                    start_ts = f"{start_h:02d}.{start_m:02d}.{start_s:06.3f}"
                    
                    end_h = int(end_seconds // 3600)
                    end_m = int((end_seconds % 3600) // 60)
                    end_s = end_seconds % 60
                    end_ts = f"{end_h:02d}.{end_m:02d}.{end_s:06.3f}"
                    
                    shot_filename = os.path.join(scene_output_path, f"{movie_id}_{scene_name_clean}_shot_{shot_idx+1}_{start_ts}-{end_ts}.mp4")
                    
                    # Create video writer with higher quality settings - using mp4
                    height, width, layers = shot_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for better quality
                    
                    out = cv2.VideoWriter(shot_filename, fourcc, 16, (width, height))
                    
                    # Write frames to video
                    for frame in shot_frames:
                        out.write(frame)  # Original frames are already in BGR format
                        
                    out.release()
                    
                    # Verify the written file
                    if os.path.exists(shot_filename):
                        cap_test = cv2.VideoCapture(shot_filename)
                        if cap_test.isOpened():
                            shot_frame_count = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap_test.release()
                            
                            if shot_frame_count > 0:
                                print(f"      Saved shot {shot_idx+1} to {shot_filename} ({shot_frame_count} frames)")
                            else:
                                print(f"      Error: Generated video file appears to be corrupted: {shot_filename}")
                        else:
                            print(f"      Error: Could not open generated video file: {shot_filename}")
                    else:
                        print(f"      Error: Could not create video file: {shot_filename}")
                else:
                    print(f"      No frames to save for shot {shot_idx+1}, skipping...")


def main():
    # Load the JSON file containing scene information
    json_path = r"F:\dataset\movie\movies_scenes.json"
    moviebench_path = r"F:\dataset\movie\moviebench"
    output_base_path = r"F:\dataset\movie\shots_output"
    
    print("Loading scene data...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    print("Loading TransNetV2 model...")
    model = TransNetV2()
    
    # Load your model weights
    state_dict = torch.load(r"F:\models\transnet-v2\transnetv2-pytorch-weights.pth")
    model.load_state_dict(state_dict)
    
    print("Starting processing...")
    process_movie_scenes(json_data, moviebench_path, output_base_path, model)
    print("Processing complete!")


if __name__ == "__main__":
    main()