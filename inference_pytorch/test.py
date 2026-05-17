import torch
import numpy as np
from transnetv2_pytorch import TransNetV2

model = TransNetV2()
state_dict = torch.load(r"F:\models\transnet-v2\transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval().cuda()

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

with torch.no_grad():
    # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
    input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    single_frame_pred, all_frame_pred = model(input_video.cuda())
    
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

# Convert predictions to actual scene/shots
single_shot_segments = predictions_to_scenes(single_frame_pred[0, :, 0])
all_frame_shot_segments = predictions_to_scenes(all_frame_pred[0, :, 0])

print("Single frame prediction shot segments (start_frame, end_frame):")
for i, (start, end) in enumerate(single_shot_segments):
    print(f"Shot {i+1}: Frame {start} to Frame {end}")

print("\nMany hot prediction shot segments (start_frame, end_frame):")
for i, (start, end) in enumerate(all_frame_shot_segments):
    print(f"Shot {i+1}: Frame {start} to Frame {end}")