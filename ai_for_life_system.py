import os
import time
import threading
from collections import deque
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision.models.video as video_models
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests

# ================= CẤU HÌNH HỆ THỐNG & AI =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = 'best_violence_model.pth'
CONFIG_FILE = 'last_ip.txt'
NTFY_TOPIC = "canhbao"

# Khởi tạo mô hình 3D-ResNet (R3D-18)
print(f"[*] Khởi tạo mô hình 3D-ResNet trên thiết bị: {DEVICE}...")
model = video_models.r3d_18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[+] Đã nạp thành công bộ trọng số huấn luyện!")
else:
    print("[!] CẢNH BÁO: Không tìm thấy best_violence_model.pth, dùng trọng số ngẫu nhiên.")
model = model.to(DEVICE)
model.eval()

# Hàm gửi cảnh báo đẩy qua mạng
def send_push_alert(frame_img, conf):
    def _send():
        try:
            cv2.imwrite("alert_snapshot.jpg", frame_img)
            headers = {
                "Title": "CANH BAO AN NINH (AI FOR LIFE)",
                "Priority": "urgent",
                "Tags": "warning"
            }
            msg = f"Phat hien hanh vi bao luc! Do tin cay: {conf:.1f}% tai {datetime.now().strftime('%H:%M:%S')}"
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=msg.encode('utf-8'),
                headers=headers,
                timeout=5
            )
            print("[+] Đã gửi thông báo đẩy ntfy thành công!")
        except Exception as e:
            print(f"[!] Lỗi kết nối ntfy: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ================= GIAO DIỆN TRẠM GIÁM SÁT HIỆN ĐẠI =================
class ModernSecurityDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI FOR LIFE - HỆ THỐNG AN NINH PHÁT HIỆN BẠO LỰC THỜI GIAN THỰC")
        self.root.geometry("1020x700")
        self.root.configure(bg="#0f172a")

        self.cap = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=NUM_FRAMES)
        self.latest_frame = None
        self.last_alert_time = 0

        self.current_state = "SAFE"
        self.current_conf = 0.0
        self.fps_display = "0.0 FPS"
        self.frame_proc_count = 0
        self.start_timer = time.time()

        self.setup_ui()

    def setup_ui(self):
        # Thanh tiêu đề phía trên
        header_frame = tk.Frame(self.root, bg="#1e293b", height=60)
        header_frame.pack(fill="x", side="top")

        title_lbl = tk.Label(
            header_frame, 
            text="HỆ THỐNG GIÁM SÁT AN NINH THÔNG MINH - AI FOR LIFE", 
            font=("Segoe UI", 14, "bold"), fg="#38bdf8", bg="#1e293b"
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        self.sys_status_lbl = tk.Label(
            header_frame, text="● HỆ THỐNG SẴN SÀNG", 
            font=("Segoe UI", 10, "bold"), fg="#10b981", bg="#1e293b"
        )
        self.sys_status_lbl.pack(side="right", padx=20)

        # Khu vực nhập cấu hình camera IP
        config_bar = tk.Frame(self.root, bg="#0f172a")
        config_bar.pack(fill="x", padx=20, pady=10)

        tk.Label(config_bar, text="Địa chỉ Camera IP:", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#0f172a").pack(side="left")

        saved_ip = "http://192.168.1.72:8080/video"
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    c = f.read().strip()
                    if c:
                        saved_ip = c
            except Exception:
                pass

        self.ip_entry = tk.Entry(config_bar, font=("Segoe UI", 10), width=36, bg="#1e293b", fg="#f8fafc", insertbackground="#38bdf8", relief="flat")
        self.ip_entry.insert(0, saved_ip)
        self.ip_entry.pack(side="left", padx=10, ipady=4)

        self.btn_toggle = tk.Button(
            config_bar, text="▶ KÍCH HOẠT QUAN SÁT", font=("Segoe UI", 10, "bold"),
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
            relief="flat", padx=15, pady=4, cursor="hand2", command=self.toggle_monitoring
        )
        self.btn_toggle.pack(side="left", padx=5)

        # Thân giao diện chia 2 cột: Trái (Khung Video), Phải (Bảng điều khiển & Nhật ký)
        body_frame = tk.Frame(self.root, bg="#0f172a")
        body_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # Cột trái: Khung hiển thị Camera
        left_col = tk.Frame(body_frame, bg="#1e293b", bd=1, relief="solid")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=5)

        cam_bar = tk.Frame(left_col, bg="#0f172a", height=30)
        cam_bar.pack(fill="x")
        tk.Label(cam_bar, text="LUỒNG CAMERA TRỰC TIẾP (LIVE STREAM)", font=("Segoe UI", 9, "bold"), fg="#64748b", bg="#0f172a").pack(side="left", padx=10, pady=4)
        self.fps_lbl = tk.Label(cam_bar, text="0.0 FPS", font=("Segoe UI", 9), fg="#64748b", bg="#0f172a")
        self.fps_lbl.pack(side="right", padx=10)

        self.video_canvas = tk.Label(left_col, bg="#020617")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Cột phải: Thông số, Trạng thái & Nhật ký sự kiện
        right_col = tk.Frame(body_frame, bg="#1e293b", width=340, bd=1, relief="solid")
        right_col.pack(side="right", fill="y", pady=5)
        right_col.pack_propagate(False)

        tk.Label(right_col, text="TRẠNG THÁI NHẬN DIỆN HIỆN TẠI", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b").pack(anchor="w", padx=15, pady=(15, 5))

        self.banner_alert = tk.Label(
            right_col, text="AN TOÀN / BÌNH THƯỜNG", font=("Segoe UI", 12, "bold"),
            bg="#065f46", fg="#ecfdf5", height=2
        )
        self.banner_alert.pack(fill="x", padx=15, pady=5)

        # Thanh hiển thị độ tin cậy
        metric_box = tk.Frame(right_col, bg="#0f172a", padx=10, pady=10)
        metric_box.pack(fill="x", padx=15, pady=8)

        tk.Label(metric_box, text="Xác suất bạo lực:", font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a").pack(anchor="w")
        self.conf_bar_canvas = tk.Canvas(metric_box, height=14, bg="#1e293b", highlightthickness=0)
        self.conf_bar_canvas.pack(fill="x", pady=5)
        self.conf_bar_fill = self.conf_bar_canvas.create_rectangle(0, 0, 0, 14, fill="#10b981", width=0)

        self.conf_text_lbl = tk.Label(metric_box, text="0.0%", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#0f172a")
        self.conf_text_lbl.pack(anchor="e")

        # Khung nhật ký sự kiện
        tk.Label(right_col, text="NHẬT KÝ SỰ KIỆN AN NINH", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b").pack(anchor="w", padx=15, pady=(15, 5))
        log_frame = tk.Frame(right_col, bg="#020617")
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.log_list = tk.Listbox(log_frame, bg="#020617", fg="#94a3b8", font=("Consolas", 8), borderwidth=0, highlightthickness=0)
        self.log_list.pack(fill="both", expand=True, padx=5, pady=5)
        self.add_log("Hệ thống khởi tạo thành công.")

    def add_log(self, text, is_alert=False):
        now = datetime.now().strftime("%H:%M:%S")
        entry = f"[{now}] {text}"
        self.log_list.insert(tk.END, entry)
        if is_alert:
            self.log_list.itemconfig(tk.END, {'fg': '#f87171'})
        self.log_list.yview(tk.END)

    def toggle_monitoring(self):
        if not self.is_running:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        cam_url = self.ip_entry.get().strip()
        try:
            with open(CONFIG_FILE, "w") as f:
                f.write(cam_url)
        except Exception:
            pass

        self.cap = cv2.VideoCapture(cam_url)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi Kết Nối", "Không thể kết nối đến Camera IP!\nVui lòng kiểm tra lại địa chỉ Wi-Fi và ứng dụng trên điện thoại.")
            return

        self.is_running = True
        self.btn_toggle.config(text="⏹ DỪNG QUAN SÁT", bg="#e11d48", activebackground="#be123c")
        self.sys_status_lbl.config(text="● ĐANG GIÁM SÁT TRỰC TIẾP", fg="#38bdf8")
        self.add_log("Đã kết nối nguồn video IP Camera.")

        self.start_timer = time.time()
        self.frame_proc_count = 0

        threading.Thread(target=self.video_stream_thread, daemon=True).start()
        threading.Thread(target=self.ai_inference_thread, daemon=True).start()

    def stop_stream(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle.config(text="▶ KÍCH HOẠT QUAN SÁT", bg="#0284c7", activebackground="#0369a1")
        self.sys_status_lbl.config(text="● ĐÃ TẠM DỪNG", fg="#64748b")
        self.banner_alert.config(text="ĐÃ DỪNG GIÁM SÁT", bg="#334155", fg="#94a3b8")
        self.add_log("Dừng giám sát camera.")

    def video_stream_thread(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Xoay 90 độ để khắc phục hiển thị dọc của camera điện thoại
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            self.latest_frame = frame

            # Đưa frame vào bộ đệm phục vụ trích xuất không gian - thời gian
            f_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            f_rgb = cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB)
            self.frame_buffer.append(f_rgb)

            # Tính toán chỉ số FPS
            self.frame_proc_count += 1
            elapsed = time.time() - self.start_timer
            if elapsed >= 1.0:
                self.fps_display = f"{self.frame_proc_count / elapsed:.1f} FPS"
                self.frame_proc_count = 0
                self.start_timer = time.time()

            # Hiển thị frame lên giao diện
            frame_disp = cv2.resize(frame, (580, 420))
            frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_disp)
            imgtk = ImageTk.PhotoImage(image=img)

            self.root.after(0, self.update_canvas, imgtk)
            time.sleep(0.016)

    def update_canvas(self, imgtk):
        if self.is_running:
            self.video_canvas.imgtk = imgtk
            self.video_canvas.configure(image=imgtk)
            self.fps_lbl.config(text=self.fps_display)

    def ai_inference_thread(self):
        while self.is_running:
            if len(self.frame_buffer) == NUM_FRAMES:
                frames_np = np.array(list(self.frame_buffer), dtype=np.float32)
                tensor_input = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
                tensor_input = tensor_input.to(DEVICE)

                with torch.no_grad():
                    outputs = model(tensor_input)
                    probs = torch.softmax(outputs, dim=1)[0]
                    v_prob = probs[1].item() * 100

                self.current_conf = v_prob
                is_violent = (v_prob >= 75.0)

                # Cập nhật bảng điều khiển an ninh
                self.root.after(0, self.update_metrics_ui, is_violent, v_prob)

                if is_violent:
                    if time.time() - self.last_alert_time > 12 and self.latest_frame is not None:
                        self.last_alert_time = time.time()
                        self.root.after(0, self.add_log, f"CẢNH BÁO: Bạo lực xác suất {v_prob:.1f}%!", True)
                        send_push_alert(self.latest_frame, v_prob)

            time.sleep(0.2)

    def update_metrics_ui(self, is_violent, conf):
        canvas_width = self.conf_bar_canvas.winfo_width() or 240
        fill_width = int((conf / 100.0) * canvas_width)

        if is_violent:
            self.banner_alert.config(text=f"🚨 PHÁT HIỆN BẠO LỰC! ({conf:.1f}%)", bg="#991b1b", fg="#fee2e2")
            self.conf_bar_canvas.itemconfig(self.conf_bar_fill, fill="#ef4444")
            self.conf_text_lbl.config(text=f"{conf:.1f}% [BÁO ĐỘNG]", fg="#f87171")
        else:
            self.banner_alert.config(text=f"AN TOÀN / BÌNH THƯỜNG ({100-conf:.1f}%)", bg="#065f46", fg="#ecfdf5")
            self.conf_bar_canvas.itemconfig(self.conf_bar_fill, fill="#10b981")
            self.conf_text_lbl.config(text=f"{conf:.1f}% [AN TOÀN]", fg="#34d399")

        self.conf_bar_canvas.coords(self.conf_bar_fill, 0, 0, fill_width, 14)

    def on_close(self):
        self.stop_stream()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = ModernSecurityDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()