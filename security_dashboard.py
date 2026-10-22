import os
import cv2
import numpy as np
import torch
import torchvision.models.video as video_models
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import winsound

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = 'best_violence_model.pth'

model = video_models.r3d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

class SecurityApp:
    def __init__(self, window):
        self.window = window
        self.window.title("HỆ THỐNG GIÁM SÁT AN NINH - AI FOR LIFE")
        self.window.geometry("750x620")
        self.window.configure(bg="#1e1e2e")

        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=NUM_FRAMES)
        self.violence_counter = 0
        self.frame_count = 0  # Bộ đếm khung hình để giảm tải cho CPU

        # Lưu kết quả dự đoán gần nhất
        self.current_label = "TRẠNG THÁI: HỆ THỐNG SẴN SÀNG"
        self.current_bg = "#313244"
        self.current_conf = 0.0

        self.setup_ui()

    def setup_ui(self):
        tk.Label(
            self.window, 
            text="HỆ THỐNG PHÁT HIỆN HÀNH VI BẠO LỰC TRỰC TIẾP", 
            font=("Segoe UI", 15, "bold"), 
            fg="#89b4fa", bg="#1e1e2e"
        ).pack(pady=8)

        self.video_frame = tk.Frame(self.window, width=560, height=315, bg="black")
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(pady=5)

        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill="both")

        self.status_box = tk.Label(
            self.window, 
            text=self.current_label, 
            font=("Segoe UI", 12, "bold"), 
            bg=self.current_bg, fg="#a6adc8", 
            width=50, height=2
        )
        self.status_box.pack(pady=8)

        self.info_label = tk.Label(
            self.window, 
            text="Độ tin cậy: --%  |  Chế độ: Đang chờ luồng dữ liệu", 
            font=("Segoe UI", 10), 
            bg="#1e1e2e", fg="#cdd6f4"
        )
        self.info_label.pack(pady=2)

        btn_frame = tk.Frame(self.window, bg="#1e1e2e")
        btn_frame.pack(pady=10)

        self.btn_webcam = tk.Button(
            btn_frame, text="📹 Bật Webcam Live", 
            font=("Segoe UI", 10, "bold"), bg="#a6e3a1", fg="#11111b", 
            padx=10, pady=5, command=self.start_webcam
        )
        self.btn_webcam.grid(row=0, column=0, padx=6)

        self.btn_file = tk.Button(
            btn_frame, text="📂 Chọn Video File", 
            font=("Segoe UI", 10, "bold"), bg="#89b4fa", fg="#11111b", 
            padx=10, pady=5, command=self.start_video_file
        )
        self.btn_file.grid(row=0, column=1, padx=6)

        self.btn_stop = tk.Button(
            btn_frame, text="⏹ Dừng", 
            font=("Segoe UI", 10, "bold"), bg="#f38ba8", fg="#11111b", 
            padx=10, pady=5, command=self.stop_stream
        )
        self.btn_stop.grid(row=0, column=2, padx=6)

    def start_webcam(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể truy cập Webcam!")
            return
        self.is_running = True
        self.frame_buffer.clear()
        self.frame_count = 0
        self.update_stream(is_live=True)

    def start_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video kiểm thử",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )
        if not file_path:
            return
        self.stop_stream()
        self.cap = cv2.VideoCapture(file_path)
        self.is_running = True
        self.frame_buffer.clear()
        self.frame_count = 0
        self.update_stream(is_live=False)

    def update_stream(self, is_live=False):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            if not is_live:
                self.status_box.config(text="ĐÃ HOÀN TẤT VIDEO", bg="#313244", fg="#a6adc8")
            self.stop_stream()
            return

        self.frame_count += 1

        # Resize nhỏ ngay từ đầu để tiết kiệm RAM & CPU
        f_small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        f_rgb = cv2.cvtColor(f_small, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(f_rgb)

        # CHỈ CHẠY DỰ ĐOÁN MỖI 8 KHUNG HÌNH (khoảng 0.25s một lần) để giao diện không bị giật lag
        if len(self.frame_buffer) == NUM_FRAMES and self.frame_count % 8 == 0:
            frames_np = np.array(list(self.frame_buffer), dtype=np.float32)
            tensor_input = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
            tensor_input = tensor_input.to(DEVICE)

            with torch.no_grad():
                outputs = model(tensor_input)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                self.current_conf = probs[pred_idx].item() * 100

            if pred_idx == 1 and self.current_conf >= 80.0:
                self.violence_counter += 1
                self.current_label = f"🚨 PHÁT HIỆN BẠO LỰC! ({self.current_conf:.1f}%)"
                self.current_bg = "#c62828"
                if self.violence_counter >= 2:
                    try:
                        winsound.Beep(1000, 80)
                    except:
                        pass
            else:
                self.violence_counter = 0
                self.current_label = f"BÌNH THƯỜNG / AN TOÀN ({self.current_conf:.1f}%)"
                self.current_bg = "#2e7d32"

            self.status_box.config(text=self.current_label, bg=self.current_bg, fg="white")
            self.info_label.config(
                text=f"Độ tin cậy: {self.current_conf:.2f}% | Chế độ: {'Webcam Live' if is_live else 'File Video'}"
            )

        # Hiển thị video mượt mà
        frame_disp = cv2.resize(frame, (560, 315))
        frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_disp)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Lặp lại sau 15ms (~60 FPS cho video display)
        self.window.after(15, lambda: self.update_stream(is_live))

    def stop_stream(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.stop_stream()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = SecurityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()