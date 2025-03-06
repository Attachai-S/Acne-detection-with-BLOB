import cv2
import dlib
import numpy as np
import imutils
from collections import OrderedDict

from src.test import shape_predictor_path

# from google.colab.patches import cv2_imshow  # ใช้สำหรับ Google Colab

# กำหนดจุด Landmark ที่ต้องการลบ
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48))
])

# โหลด Dlib's face detector และ Landmark predictor
shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat"
# shape_predictor_path = r"src\shape_predictor_68_face_landmarks.dat"
image_path = r"src\result\extracted_facial\cropped_face_0.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# อ่านและแปลงภาพเป็นขาวดำ
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
rects = detector(gray, 1)

# สร้าง mask สีดำขนาดเท่ากับภาพต้นฉบับ
mask = np.ones(image.shape[:2], dtype="uint8") * 255  # ค่า 255 คือเก็บส่วนที่ต้องการไว้

def shape_to_numpy_array(shape, dtype="int"):
    """แปลง Dlib Shape Object เป็น NumPy Array"""
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates

# วนลูปผ่านใบหน้าที่ตรวจพบ
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

    # ลบดวงตาและปากโดยวาดเป็นสีดำบน Mask
    for (name, (j, k)) in FACIAL_LANDMARKS_INDEXES.items():
        pts = shape[j:k]
        hull = cv2.convexHull(pts)
        cv2.drawContours(mask, [hull], -1, 0, -1)  # วาดสีดำในจุดที่ต้องการลบ

# ใช้ Mask เพื่อลบดวงตาและปากออกจากภาพ
result = cv2.bitwise_and(image, image, mask=mask)


# แสดงผลลัพธ์
cv2.imshow(result)
cv2.waitKey(0)
# cv2.destroyAllWindows()
# output_path = f"marked_face_{i}.jpg"
# cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))