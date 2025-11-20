import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)  


background = None
for i in range(50):
    ret, background = cap.read()

background = cv2.flip(background, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

  
    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow("Invisible Cloak", final_output)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Green color range
#     lower_green = np.array([35, 100, 100])
#     upper_green = np.array([85, 255, 255])

#     # Create the green mask
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Clean the mask
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

#     mask_inv = cv2.bitwise_not(mask)

#     # Background part (only where cloak is present)
#     res1 = cv2.bitwise_and(background, background, mask=mask)

#     # Real frame without cloak region
#     res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

#     final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

#     cv2.imshow("Invisible Cloak - Green", final_output)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)
# background = None

# # Capture the background (first few frames)
# for i in range(60):
#     ret, background = cap.read()
#     background = cv2.flip(background, 1)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # ALL GREEN COLOR RANGE
#     lower_green = np.array([25, 40, 40])     # very dark green
#     upper_green = np.array([95, 255, 255])   # very bright/neon green

#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Clean the mask
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

#     mask_inv = cv2.bitwise_not(mask)

#     # Replace green region with background
#     res1 = cv2.bitwise_and(background, background, mask=mask)

#     # Keep rest of frame normal
#     res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

#     final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

#     cv2.imshow("Invisible Cloak - All Green Colors", final_output)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()