#BLOB Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

# face = cv2.imread('face_segmented.jpg')
img = r"C:\cygwin64\home\vangu\Acne-detection-with-BLOB\src\result\marked_facial\marked_face.jpg"
face = cv2.imread(img)
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