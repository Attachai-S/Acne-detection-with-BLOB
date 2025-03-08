import os
import cv2

folder_path = r"images"  # เปลี่ยนเป็นพาธโฟลเดอร์ของคุณ

# อ่านไฟล์ทั้งหมดในโฟลเดอร์
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# วนลูปอ่านไฟล์ภาพ
for i,image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error loading {image_path}")
        continue

    print(f"✅ Loaded: {image_path},num {i+1}")
    cv2.imshow("Image", image)
    cv2.waitKey(500)  # แสดงภาพ 0.5 วินาที
    cv2.destroyAllWindows()
