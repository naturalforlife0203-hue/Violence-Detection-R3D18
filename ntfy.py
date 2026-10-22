import requests
import cv2

# Tên topic 
NTFY_TOPIC = "canhbao"

def send_push_alert(frame_img, conf):
    # Lưu ảnh hiện trường
    cv2.imwrite("alert_snapshot.jpg", frame_img)
    
    # Gửi thông báo đẩy kèm tiêu đề và rung chuông khẩn cấp
    headers = {
        "Title": "CANH BAO AN NINH (AI FOR LIFE)",
        "Priority": "urgent",  # Mức độ khẩn cấp (rung chuông to)
        "Tags": "warning,rotating_light"
    }
    msg = f"Phat hien hanh vi bao luc! Do tin cay: {conf:.1f}%"
    
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=msg.encode('utf-8'),
            headers=headers,
            timeout=5
        )
        print("[+] Đã gửi thông báo đẩy đến điện thoại!")
    except Exception as e:
        print(f"[!] Lỗi gửi ntfy: {e}")