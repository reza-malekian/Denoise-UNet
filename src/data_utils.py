import os
import cv2
import numpy as np

# تابع خواندن تصاویر از فولدر
def load_images_from_folder(folder, size=(128,128), grayscale=False, max_images=None):
    imgs = []
    files = sorted(os.listdir(folder))
    if max_images:
        files = files[:max_images]
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR if not grayscale else cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0  # نرمال‌سازی به بازه 0-1
        imgs.append(img)
    return np.array(imgs)

# بارگذاری تصاویر
train_folder = 'data/train'
val_folder = 'data/validation'

x_train = load_images_from_folder(train_folder)
x_val = load_images_from_folder(val_folder)

print("Train shape:", x_train.shape)
print("Validation shape:", x_val.shape)
