import os
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
import numpy as np
import dlib
import imutils
from collections import OrderedDict
import time

# folder_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\images"
# output_MTCNN_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\extracted_facial"
''' Important path '''
folder_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\images"
output_MTCNN_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\extracted_facial"
shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat" #for dlib

output_dlib_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial"
output_blob_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\final_result"
''' Important path '''


# สร้างโฟลเดอร์ output ถ้ายังไม่มี
os.makedirs(output_MTCNN_path, exist_ok=True)

# อ่านไฟล์ทั้งหมดในโฟลเดอร์
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# วนลูปอ่านไฟล์ภาพ
for runnumber, image_file in enumerate(image_files):

    '''output path'''
    image_path = os.path.join(folder_path, image_file)
    output_extracted_facial = os.path.join(output_MTCNN_path, f"cropped_{image_file}")
    output_marked_facial = os.path.join(output_dlib_path, f"marked_face_{image_file}")
    '''output path'''

    # อ่านภาพ
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    detections = detector.detect_faces(image_rgb)

    for i, face in enumerate(detections):
        x, y, w, h = face['box']  # bounding box
        cropped_face = image_rgb[y:y + h, x:x + w]  # ตัดเฉพาะส่วนของใบหน้า

        cv2.imwrite(output_extracted_facial, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

    # แปลงเป็น Grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # บันทึกภาพที่แปลงแล้ว
    # cv2.imwrite(output_path, gray_image)

    print(f"✅ Saved: {output_extracted_facial}, num {runnumber+1}")

    ''' dlib '''
    FACIAL_LANDMARKS_INDEXES = OrderedDict([
        ("Mouth", (48, 68)),
        ("Right_Eyebrow", (17, 22)),
        ("Left_Eyebrow", (22, 27)),
        ("Right_Eye", (36, 42)),
        ("Left_Eye", (42, 48))
    ])
    # โหลด Dlib's face detector และ Landmark predictor
    # shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat"
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
    cv2.imwrite(output_marked_facial, cv2.cvtColor(dlib_result, cv2.COLOR_RGB2BGR))
    print(f"☑️ Saved: {output_marked_facial}, num {runnumber + 1}")
    ''' dlib '''

    ''' BLOB '''
marked_folder = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial"
marked_image_files = [f for f in os.listdir(marked_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

for runnumber, image_file in enumerate(marked_image_files):

    image_path = os.path.join(marked_folder, image_file)
    # output_marked_facial = os.path.join(output_dlib_path, f"marked_face_{image_file}")
    output_final_result = os.path.join(output_blob_path, f"final_result_{image_file}")

    image = cv2.imread(image_path)

    ''' BLOB '''
    gray_segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoise with Gaussian Blur
    blurred = cv2.GaussianBlur(gray_segmented, (5, 5), 0)

    # กำหนด Parameters สำหรับ BLOB Detector
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 10  # ขนาด BLOB ขั้นต่ำ (min "detect every = plx")
    params.maxArea = 1500  # ขนาด BLOB สูงสุด (max "--")

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
    image_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imwrite(blob_path, image_with_blobs)
    cv2.imwrite(output_final_result, image_with_blobs)


    print(f"iamge {runnumber} acne spot detected "
          f": {len(keypoints)}")

    # for i, kp in enumerate(keypoints):
    #     x, y = kp.pt  # ตำแหน่งจุด (X, Y)
    #     size = kp.size  # ขนาดของจุด
    #     print(f"สิวที่ {i + 1}: ตำแหน่ง ({x:.2f}, {y:.2f}), ขนาด {size:.2f} pixels")
    ''' BLOB '''
#end program
