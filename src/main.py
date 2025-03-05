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

result_path = "result/extracted_facial"
if not os.path.exists(result_path):
    os.makedirs(result_path)


# โหลด MTCNN detector
detector = MTCNN()

# อ่านรูปภาพ
image_path = "images/acne5.jpg"  # เปลี่ยนเป็น path ของรูปภาพ
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB

# ตรวจจับใบหน้า
detections = detector.detect_faces(image_rgb)

# แสดงผลลัพธ์
for face in detections:
    x, y, width, height = face['box']
    keypoints = face['keypoints']

    # วาดกรอบใบหน้า
    cv2.rectangle(image_rgb, (x, y), (x+width, y+height), (0, 255, 0), 2)

    # วาด facial landmarks
    for key, point in keypoints.items():
        cv2.circle(image_rgb, point, 2, (255, 0, 0), 2)

# วนลูปผ่านทุกใบหน้าที่ตรวจพบ
for i, face in enumerate(detections):
    x, y, w, h = face['box']  # ดึง bounding box
    cropped_face = image_rgb[y:y+h, x:x+w]  # ตัดเฉพาะส่วนของใบหน้า

    # บันทึกหรือแสดงผลใบหน้าที่ตัดออกมา
    plt.imshow(cropped_face)
    plt.axis("off")
    plt.show()

    # บันทึกไฟล์
    output_path = f"result/extracted_facial/cropped_face_{i}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

