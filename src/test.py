import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN  # ตรวจสอบว่า mtcnn ติดตั้งแล้ว (pip install mtcnn)
import os
import dlib
import imutils
from collections import OrderedDict
import numpy as np

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

# shape_predictor_path = "\src\shape_predictor_68_face_landmarks.dat"  # Update with the correct path

facial_features_cordinates = {}

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    # ("Right_Eyebrow", (17, 22)),
    # ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    # ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

# Instead, directly define the paths:
shape_predictor_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\shape_predictor_68_face_landmarks.dat"  # Replace with your actual path
image_path = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\extracted_facial\cropped_face_0.jpg"


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    print(facial_features_cordinates)
    return output

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path) # Use the defined path

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(image_path) # Use the defined path
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)
    output = visualize_facial_landmarks(image, shape)
    # cv2.imshow(output)
    # cv2.imshow(output, image)# Use cv2_imshow instead of cv2.imshow
    # cv2.waitKey(0)
    axes[0].imshow(output)
    axes[0].set_title("Final Image")
    axes[0].axis("off")
    plt.show()
    # output_path = f"{output}/final{i}.jpg"
    output_path = f"{output_folder}/final_{i}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

