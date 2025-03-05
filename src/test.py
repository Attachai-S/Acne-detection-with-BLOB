import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN  # ตรวจสอบว่า mtcnn ติดตั้งแล้ว (pip install mtcnn)
import os

# โหลดรูปภาพ
image_path = "images/acne5.jpg"  # เปลี่ยนเป็น path ของรูปภาพ
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB

# ตรวจจับใบหน้า
detector = MTCNN()
detections = detector.detect_faces(image_rgb)

# แสดงผลลัพธ์
fig, axes = plt.subplots(1, len(detections) + 1, figsize=(5 * (len(detections) + 1), 5))

# แสดงภาพต้นฉบับในช่องแรก
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# ตรวจสอบว่ามีโฟลเดอร์สำหรับเซฟผลลัพธ์หรือไม่
output_folder = "result/extracted_facial"
os.makedirs(output_folder, exist_ok=True)

# วนลูปผ่านทุกใบหน้าที่ตรวจพบ
for i, face in enumerate(detections):
    x, y, w, h = face['box']  # ดึง bounding box
    cropped_face = image_rgb[y:y+h, x:x+w]  # ตัดเฉพาะส่วนของใบหน้า

    # แสดงภาพใบหน้าที่ถูกตัด
    axes[i + 1].imshow(cropped_face)
    axes[i + 1].set_title(f"Extracted_facial")
    axes[i + 1].axis("off")

    # บันทึกไฟล์
    output_path = f"{output_folder}/cropped_face_{i}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

# แสดงผลทั้งหมด
plt.show()
