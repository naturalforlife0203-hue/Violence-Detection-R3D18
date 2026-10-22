import os
import shutil
import random

# Thiết lập tỷ lệ chia: 70% Train, 15% Val, 15% Test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = 'raw_data'
OUT_DIR = 'dataset'
CLASSES = ['Fight', 'NonFight']
SPLITS = ['train', 'val', 'test']

# Tạo cấu trúc thư mục đích
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

# Tiến hành phân chia file
random.seed(42)
for cls in CLASSES:
    src_folder = os.path.join(RAW_DIR, cls)
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.mp4', '.avi'))]
    random.shuffle(files)
    
    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    for f in train_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(OUT_DIR, 'train', cls, f))
    for f in val_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(OUT_DIR, 'val', cls, f))
    for f in test_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(OUT_DIR, 'test', cls, f))

    print(f"Lớp {cls}: Tổng {total} video -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

print("Hoàn tất chia tập dữ liệu!")