import os
import cv2
import numpy as np
import torch
import torchvision.models.video as video_models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
IMG_SIZE = 112
CLASSES = ['NonFight', 'Fight']

# Load model
model = video_models.r3d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_violence_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict_video(video_path):
    if not os.path.exists(video_path):
        print(f"[LOI] Khong tim thay file: {video_path}")
        return None, 0.0

    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // NUM_FRAMES)
    count = 0
    
    while cap.isOpened() and len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        count += 1
    cap.release()

    if len(frames) == 0:
        return None, 0.0

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    # Dung np.array() truoc de het canh bao UserWarning
    frames_np = np.array(frames, dtype=np.float32)
    tensor_input = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    tensor_input = tensor_input.to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return pred_idx, probs[pred_idx].item() * 100

if __name__ == '__main__':
    # 1. Test thu 5 video co that trong dataset/test/Fight
    fight_test_dir = 'dataset/test/Fight'
    fight_files = os.listdir(fight_test_dir)[:5]
    print("=== DU DOAN THU 5 VIDEO BAO LUC TRONG TAP TEST ===")
    for f in fight_files:
        path = os.path.join(fight_test_dir, f)
        pred, conf = predict_video(path)
        print(f"Video: {f:15s} -> Doan: {CLASSES[pred]:10s} ({conf:.2f}%)")

    # 2. Test thu 5 video co that trong dataset/test/NonFight
    nonfight_test_dir = 'dataset/test/NonFight'
    nonfight_files = os.listdir(nonfight_test_dir)[:5]
    print("\n=== DU DOAN THU 5 VIDEO BINH THUONG TRONG TAP TEST ===")
    for f in nonfight_files:
        path = os.path.join(nonfight_test_dir, f)
        pred, conf = predict_video(path)
        print(f"Video: {f:15s} -> Doan: {CLASSES[pred]:10s} ({conf:.2f}%)")