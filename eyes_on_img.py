import cv2
import dlib
import numpy as np



THRESHOLD_VALUE = 100


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def pixels_b2w(img):
    mask = (img == [0, 0, 0]).all(axis=2)
    img[mask] = [255, 255, 255]
    return img


def eyes_on_mask(shape, img_size, eyes_points, dilate=False):

    mask = np.zeros(img_size, dtype=np.uint8)

    for eye_points in eyes_points:
        points = [shape[i] for i in eye_points]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)

    if dilate:
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, 5)

    return mask


def thresholding(img, threshold_value):
    _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    #thresh = cv2.erode(thresh, None, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=4)
    #thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh) # per trovare il contorno lo sfondo deve essere nero e l'oggetto bianco
    return thresh


def contouring(thresh, mid, img, side):

    for _ in range(2):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if side.lower()=="r":
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        except:
            pass


def _rect2bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

img = cv2.imread("resources/eyes_left.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for rect in rects:

    (x, y, w, h) = _rect2bb(rect)

    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imwrite("img_rect.jpg", img)

    shape = predictor(gray, rect) # eseguiamo il landmarks detection
    shape = shape_to_np(shape) # convertiamo in un array numpy

    """"
    for i in range(68):
        cv2.circle(img, (int(shape[i][0]), int(shape[i][1])), 4, (0, 255, 0), -1)
        cv2.imwrite("img_landmarks.jpg", img)

    for p in zip(_EYE_LEFT, _EYE_RIGHT):
        pl, pr = p
        cv2.circle(img, (int(shape[pl][0]), int(shape[pl][1])), 3, (0, 255, 0), -1)
        cv2.circle(img, (int(shape[pr][0]), int(shape[pr][1])), 3, (0, 255, 0), -1)
        cv2.imwrite("img_landmarks_eyes.jpg", img)
    """

    mask = eyes_on_mask(shape, img.shape[:2], (_EYE_RIGHT, _EYE_LEFT), dilate=True)
    #cv2.imwrite("img_eyes_on_mask.jpg", mask)

    eyes = cv2.bitwise_and(img, img, mask=mask) # segmentiamo gli occhi dall'immagine
    #cv2.imwrite("img_mask_on_eyes.jpg", eyes)

    # rendiamo tutti i pixel neri bianchi, così solo la pupilla rimane nera
    eyes = pixels_b2w(eyes)
    #cv2.imwrite("img_mask_on_eyes_black.jpg", eyes)

    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY) #... però dobbiamo convertire in bw
    thresh = thresholding(eyes_gray, THRESHOLD_VALUE)

    cv2.imwrite("img_eyes_gray.jpg", eyes_gray)
    cv2.imwrite("img_tresg.jpg", thresh)

    mid = (shape[42][0] + shape[39][0]) // 2
    contouring(thresh[:, 0:mid], mid, img, side="L")
    contouring(thresh[:, mid:], mid, img, side="R")


cv2.imshow('mask', mask.astype(np.uint8)*255)
cv2.imshow('eyes', eyes)
cv2.imshow('img', img)
cv2.imshow("tresh", thresh)

cv2.imwrite('img.jpg',img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
