import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import matplotlib.pyplot as plt

# 1. Cấu hình phần cứng và siêu tham số
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
IMG_SIZE = 112
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0001

# 2. Xây dựng Dataset nạp 16 frames từ video
class ViolenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = ['NonFight', 'Fight']
        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.exists(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.mp4', '.avi')):
                    self.samples.append((os.path.join(cls_folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
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
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            count += 1
        cap.release()

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=torch.uint8).numpy())

        tensor_frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        return tensor_frames, torch.tensor(label, dtype=torch.long)

# 3. Hàm chạy chính (bắt buộc trên Windows)
def main():
    print(f"--> Đang sử dụng thiết bị: {DEVICE}")

    train_set = ViolenceDataset('dataset/train')
    val_set = ViolenceDataset('dataset/val')

    # num_workers=0 để tránh crash multiprocessing trên Windows
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"--> Tổng mẫu Train: {len(train_set)}, Mẫu Val: {len(val_set)}")

    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    print("\n=== BẮT ĐẦU HUẤN LUYỆN ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_set)
        epoch_acc = correct.double() / len(train_set)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_set)
        epoch_val_acc = val_correct.double() / len(val_set)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f}")

    torch.save(model.state_dict(), 'best_violence_model.pth')
    print("\n--> Đã lưu trọng số mô hình vào file 'best_violence_model.pth'")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_chart.png')
    print("--> Đã xuất biểu đồ vào file 'loss_chart.png'")

if __name__ == '__main__':
    main()