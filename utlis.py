import cv2
import numpy as np


def getContours(img, cThresh=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThresh[0], cThresh[1])

    # apply dilate and erode to get good edges
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    if showCanny:
        cv2.imshow("Canny", imgThre)

    # find the contour and store
    contours, hierachy = cv2.findContours(
        imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    finalContours = []

    # loop though contour to find the area. all this will give us the area of close curve
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:  # user to define minArea
            # find parameter length, True = close
            peri = cv2.arcLength(i, True)
            # find corner points
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # find bounding box
            bbox = cv2.boundingRect(approx)

            # we only want rectangle and exclude others line. So filter any points.
            # if user insert filter value, then we will filter based on that value and append to empty list based on the filter value, else append every approx
            if filter > 0:
                if len(approx) == filter:
                    # if corner points == filter, then append for info in () into the emptylist
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    # sort contours based on size area, area is in index 1 of finalContours [], key need to be in a function, so we use lambda function
    # lambda function used only when we wanna use that function once, application in sorting & filtering data, reverse=true meaning sort in DECENDING order
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    # draw contour that were detected
    if draw:
        for con in finalContours:
            # (img, contour in 4th element,)
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


# contours points might not be in correct orientation, it can be mixed 1,2,3,4 CCW or 2,3,1,4 etc.
# so we need to sort it to TL,TR,BL,BR = 1,2,3,4
# TL =1 should be the smallest, BR = 4 should be the biggest, 2 should bigger that 1


def reorder(myPoints):
    print(myPoints.shape)
    # since received data as shape (4,1,2), need to send back after add with the same shape
    myPointsNew = np.zeros_like(myPoints)
    # shape (4,1,2), points have 4 dimensions with each dimension has 2 element, x and y, "1" is redundant. so reshape.
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


# warpImg is actually our A4 paper
def warpImg(img, points, w, h, pad=20):
    # print(points)
    # print(reorder(points))
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))

    # use padding to remove extra corner paper
    imgWarp = imgWarp[pad : imgWarp.shape[0] - pad, pad : imgWarp.shape[1] - pad]

    return imgWarp


# triangulation calculation for side measurement distance = sqrt ((x2-x1)**2 + (y2-y1)**2)
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
