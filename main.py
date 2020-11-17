import cv2
import numpy as np
import utlis

###################################

webcam = True
path = "image_01.jpg"
cap = cv2.VideoCapture(0)
# set brightness
cap.set(10, 160)
# set width
cap.set(3, 1920)
# set height
cap.set(4, 1080)
# scale factor
scale = 3
# define width height A4 paper
wP = 210 * scale
hP = 297 * scale

######################################

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # img, finalContours = utlis.getContours(img, showCanny=True, draw=True)
    imgContours, finalContours = utlis.getContours(img, showCanny=False, minArea=50000, filter=4)

    if len(finalContours) != 0:
        # we already sort the biggest area at the top of the list (see utils reverse = true), and we need the approx 4 points
        biggest = finalContours[0][2]
        # print(biggest)
        # now we get the 4 corner points, need to work with those points with a new function in utils
        imgWarp = utlis.warpImg(img, biggest, wP, hP)
        # img, finalContours = utlis.getContours(img, showCanny=False, minArea=50000, filter=4
        imgContours2, finalContours2 = utlis.getContours(
            imgWarp,
            showCanny=False,
            minArea=2000,
            filter=4,
            cThresh=[50, 50],
            draw=False,
        )
        # draw polyline properly
        if len(finalContours2) != 0:
            for obj in finalContours2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                # reorder points
                nPoints = utlis.reorder(obj[2])
                # findDis(pts1, pts2) in cm (devide by 10) and 1 decimal points as new width and new height
                nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale)/ 10),1)
                nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale)/ 10),1)
                
                #draw arrow
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                
                #put text
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow("A4", imgContours2)

        # take the corner points of contours and measure

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
