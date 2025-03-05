import os
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN

# img = cv2.imread("images/acne2.jpg")
# grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(img);plt.axis('off');plt.title('Original Image')
#
# plt.subplot(1,2,2)
# plt.imshow(grey, cmap='gray')
# plt.axis('off');plt.title('Grayscale Image')
#
# plt.show()
# check and create folder 
result_path = "result/extracted_facial"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# โหลด MTCNN detector
# load image
image_path = "images/acne5.jpg" 
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# extract facial
detector = MTCNN()
detections = detector.detect_faces(image_rgb)

# show result
fig, axes = plt.subplots(1, len(detections) + 1, figsize=(5 * (len(detections) + 1), 5))

# show original image 
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# ตรวจสอบว่ามีโฟลเดอร์สำหรับเซฟผลลัพธ์หรือไม่
# output_folder = "result/extracted_facial"
# os.makedirs(output_folder, exist_ok=True)

# วนลูปผ่านทุกใบหน้าที่ตรวจพบ
for i, face in enumerate(detections):
    x, y, w, h = face['box']  # ดึง bounding box
    cropped_face = image_rgb[y:y+h, x:x+w]  # ตัดเฉพาะส่วนของใบหน้า

    # แสดงภาพใบหน้าที่ถูกตัด
    axes[i + 1].imshow(cropped_face)
    axes[i + 1].set_title(f"Extracted_facial")
    axes[i + 1].axis("off")

    # บันทึกไฟล์
    output_path = f"result/extracted_facial/cropped_face_{i}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

# แสดงผลทั้งหมด
plt.show()
