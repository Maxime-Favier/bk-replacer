import cv2
import numpy as np
from PIL import Image

# == get photo =======================================================================
camera = cv2.VideoCapture(0)

# input("press enter to continue")
_, img = camera.read()
del (camera)

# == Processing =======================================================================
BLUR = 25
CANNY_THRESH_1 = 13
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 11  
MASK_ERODE_ITER = 10
MASK_COLOR = (0, 0, 1.0)  # In BGR format

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2, L2gradient=True)
edges = cv2.dilate(edges,None ,iterations=3)
#edges = cv2.erode(edges, None, iterations=7)

# Find contours in edges, sort by area
contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))

contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

# Create empty mask, draw filled polygon on it corresponding to largest contour
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

# Smooth mask, then blur it
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

# Blend masked img into MASK_COLOR background
mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
img = img.astype('float32') / 255.0  # for easy blending
masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
# split image into channels
c_red, c_green, c_blue = cv2.split(img)
# merge with mask got on one of a previous steps
img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))



cv2.imwrite('final.png', img_a * 255)
# == Merging =======================================================================

bk = Image.open("bk.png").convert("RGBA")
photo = Image.open("final.png").convert("RGBA")

out = Image.new("RGBA", bk.size)
out = Image.alpha_composite(out, bk)
out = Image.alpha_composite(out, photo)

cv2.imshow("1", edges)
cv2.imshow('3', img)
cv2.imshow("2", mask)

out.show()
cv2.waitKey()
