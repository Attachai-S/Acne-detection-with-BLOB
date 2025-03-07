import os
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
import numpy as np
import dlib
import imutils
from collections import OrderedDict
'''
this is for single image processing by step:
MTCNN (Region of interest) -> mark facial feature with dlib -> acne detection with BLOB  
'''

''' create important directory for result'''
extract_path, marked_path, final_result_path = ("result/extracted_facial", "result/marked_facial",
                                                "result/final_result")
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
if not os.path.exists(marked_path):
    os.makedirs(marked_path)
if not os.path.exists(final_result_path):
    os.makedirs(final_result_path)

''' Important path '''
image_path = "images/acne6.jpg"
output_MTCNN_path = f"result/extracted_facial/cropped_face.jpg"
shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat" #for dlib
output_dlib_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial\marked_face.jpg"
''' Important path '''

''' MTCNN '''
# โหลด MTCNN detector
# load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# extract facial by MTCNN
detector = MTCNN()
detections = detector.detect_faces(image_rgb)

# show result
fig, axes = plt.subplots(1, len(detections) + 1, figsize=(5 * (len(detections) + 1), 5))

# show original image 
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# วนลูปผ่านทุกใบหน้าที่ตรวจพบ
for i, face in enumerate(detections):
    x, y, w, h = face['box']  #bounding box
    cropped_face = image_rgb[y:y+h, x:x+w]  # ตัดเฉพาะส่วนของใบหน้า

    # แสดงภาพใบหน้าที่ถูกตัด
    axes[i + 1].imshow(cropped_face)
    axes[i + 1].set_title(f"Extracted_facial")
    axes[i + 1].axis("off")

    # บันทึกไฟล์
    output_MTCNN_path = f"result/extracted_facial/cropped_face.jpg"
    cv2.imwrite(output_MTCNN_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

# show MTCNN result
# plt.show() #show extract facial
''' MTCNN '''

''' dlib '''
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48))
])

# โหลด Dlib's face detector และ Landmark predictor
shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat"
# image_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\extracted_facial\cropped_face.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# อ่านและแปลงภาพเป็นขาวดำ
# image = cv2.imread(cropped_face)
cropped_face = imutils.resize(cropped_face, width=500)
gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
rects = detector(gray, 1)

# สร้าง mask สีดำขนาดเท่ากับภาพต้นฉบับ
mask = np.ones(cropped_face.shape[:2], dtype="uint8") * 255  # ค่า 255 คือเก็บส่วนที่ต้องการไว้

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

# Mask to original image
dlib_result = cv2.bitwise_and(cropped_face, cropped_face, mask=mask)

# output_dlib_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial\marked_face.jpg"
cv2.imwrite(output_dlib_path, cv2.cvtColor(dlib_result, cv2.COLOR_RGB2BGR))
''' dlib '''

''' BLOB '''
#BLOB Detection
# img = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial\marked_face.jpg"
face = cv2.imread(output_dlib_path)
# #read from GRABCUT segment
gray_segmented = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# Denoise with Gaussian Blur
blurred = cv2.GaussianBlur(gray_segmented, (5,5), 0)

# กำหนด Parameters สำหรับ BLOB Detector
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 10  # ขนาด BLOB ขั้นต่ำ (min "detect every = plx")
params.maxArea = 2000  # ขนาด BLOB สูงสุด (max "--")

params.filterByCircularity = True
params.minCircularity = 0.2  # ให้ BLOB มีความกลมบางระดับ

params.filterByConvexity = True
params.minConvexity = 0.5  # ปรับค่าความนูนของ BLOB

params.filterByInertia = True
params.minInertiaRatio = 0.1  # ค่าความกลมของวัตถุ

# สร้าง BLOB Detector
detector = cv2.SimpleBlobDetector_create(params)

# ค้นหา Keypoints (ตำแหน่งของ BLOB)
keypoints = detector.detect(blurred)

# วาดจุด BLOB บนภาพต้นฉบับ
image_with_blobs = cv2.drawKeypoints(face, keypoints, np.array([]), (0, 255, 0),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# แสดงผลลัพธ์
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
axes[1].set_title("BLOB Detected (Acne Spots)")
axes[1].axis("off")
o_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\final_result\final.jpg"
cv2.imwrite(o_path, cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
plt.show()

print(f"\nจำนวนสิวที่ตรวจพบ: {len(keypoints)}")

for i, kp in enumerate(keypoints):
     x, y = kp.pt  # ตำแหน่งจุด (X, Y)
     size = kp.size  # ขนาดของจุด
     print(f"สิวที่ {i+1}: ตำแหน่ง ({x:.2f}, {y:.2f}), ขนาด {size:.2f} pixels")
''' BLOB '''