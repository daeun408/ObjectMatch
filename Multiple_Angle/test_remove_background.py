import cv2
from rembg import remove
from PIL import Image
import numpy as np
# 이미지를 로드합니다.

input = Image.open('./data/storedData/ace/ace1.jpg') # load image
output = remove(input) # remove background
#output.save('rembg.PNG') # save image

"""
# OpenCV를 사용하여 이미지를 윈도우에 표시
cv2.imshow('Output Image', cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# PIL Image를 OpenCV Image로 변환합니다.
input_cv_img = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
output_cv_img = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2BGRA)

# 원본 이미지를 화면에 표시합니다.
cv2.imshow("Original Image", input_cv_img)
cv2.waitKey(0)

# 배경이 제거된 이미지를 화면에 표시합니다.
cv2.imshow("Image with Background Removed", output_cv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()