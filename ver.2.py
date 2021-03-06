import numpy as np
import cv2

def points(pts):
    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis =1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def scanImage():
    image = cv2.imread('img_04.jpg')
    origin = image.copy()

    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edged = cv2.Canny(gray, 75, 200)

    #Edge

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    (cnts, _)= cv2.findContours(edged.copy(), cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    #Contours 

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline",image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    rect = points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    
    #회전변환 좌표 수정
    dst = np.float32([[0,0], [0, maxHeight-1], [maxWidth-1, maxHeight-1],[maxWidth-1, 0] ])
    #dst = np.float32([[0,0], [maxHeight-1, 0], [maxHeight-1,maxWidth-1], [0, maxWidth-1]])
    

    N = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(origin, N, (maxWidth, maxHeight))


    #transform
    
    cv2.imshow("Warped", warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    #Adaptive Threshold
    cv2.imshow("Original", origin)
    cv2.imshow("Scanned", warped)
    cv2.imwrite('scannedImage.png', warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    scanImage()

























    
