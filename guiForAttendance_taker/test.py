import cv2
import os
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imgBackground = cv2.imread('resources/background.png')
if imgBackground is None:
    print("Error: Could not load background image. Check the file path.")
else:
    print("Background image loaded successfully.")

folderModePath = 'resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
print(len(imgModeList))

while True:
    success, img = cap.read()
    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44+633, 808:808 +414] = imgModeList[1]
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)