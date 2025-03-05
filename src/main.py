# import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("images/acne2.jpg")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img);plt.axis('off');plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(grey, cmap='gray')
plt.axis('off');plt.title('Grayscale Image')

plt.show()