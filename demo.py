import os
import cv2
import numpy as np
import torch
import torchvision.models.video as video_models
from tkinter import Tk, filedialog

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
IMG_SIZE = 112
CLASSES = ['NonFight (Binh thuong)', 'Fight (BAO LUC)']

model = video_models.r3d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_violence_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict_single_video(video_path):
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
        print("Loi doc video!")
        return

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    frames_np = np.array(frames, dtype=np.float32)
    tensor_input = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    tensor_input = tensor_input.to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        conf = probs[pred_idx].item() * 100

    print("=" * 45)
    print(f"File: {os.path.basename(video_path)}")
    print(f"Ket qua: {CLASSES[pred_idx]}")
    print(f"Do tin cay: {conf:.2f}%")
    print("=" * 45)

if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    while True:
        print("\n--> Dang mo hop thoai chon video (Bam Cancel de thoat)...")
        file_path = filedialog.askopenfilename(
            title="Chon 1 video bat ky de kiem thu",
            filetypes=[("Video files", "*.mp4 *.avi")]
        )
        if not file_path:
            print("Da thoat chuong trinh demo.")
            break
        predict_single_video(file_path)