import cv2
import dlib
import numpy as np
import math
from audio import Audio


class Camera:

    _EYE_LEFT = [36, 37, 38, 39, 40, 41]
    _EYE_RIGHT = [42, 43, 44, 45, 46, 47]
    _THRESHOLD_VALUE = 120
    _TRACKING_SENSITIVITY = 5
    GAZE_LEFT = -1
    GAZE_RIGHT = 1
    _SKIP_PAUSE = 20
    _FACE_RECT = (160, 0, 460, 480)

    skip_for = 0

    def __init__(self):
        self._video_capture = cv2.VideoCapture(1)
        self._detector = dlib.get_frontal_face_detector()

        # link al modello: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self._predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


    def _rect2bb(self, rect):

        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        return (x, y, w, h)


    def _shape2np(self, shape):

        coords = np.zeros((68, 2))

        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords


    def pixels_b2w(self, img):
        mask = (img == [0, 0, 0]).all(axis=2)
        img[mask] = [255, 255, 255]
        return img


    def eyes_on_mask(self, shape, img_size, eyes_points, dilate=False):

        mask = np.zeros(img_size, dtype=np.uint8)

        for eye_points in eyes_points:
            points = [shape[i] for i in eye_points]
            points = np.array(points, dtype=np.int32)
            mask = cv2.fillConvexPoly(mask, points, 255)

        if dilate:
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(mask, kernel, 5)

        return mask


    def thresholding(self, img, threshold_value):
        _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)  # per trovare il contorno lo sfondo deve essere nero e l'oggetto bianco
        return thresh


    def contouring(self, thresh, mid, img, side):

        for _ in range(2):
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # otteniamo il contorno
            try:
                cnt = max(cnts, key=cv2.contourArea)
                M = cv2.moments(cnt) # calcoliamo il centro del contorno
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # se è la parte destra dell'immagine, trasliamo il centro
                if side.lower() == "r":
                    cx += mid
                return cx, cy
            except:
                pass


    def _get_eyes_center(self, shape, eyes_points):

        eyes_centers = []

        for eye_points in eyes_points:
            x = int((shape[eye_points[0]][0]+shape[eye_points[3]][0])//2)
            yt = int((shape[eye_points[1]][1]+shape[eye_points[2]][1])//2)
            yb = int((shape[eye_points[4]][1]+shape[eye_points[5]][1])//2)
            y = int((yt+yb)//2)
            eyes_centers.append((x,y))

        return eyes_centers

    def _track_eyes_movements(self, eyes_pos, iris_pos):

        # calcoliamo la distanza tra centro dell'occhio e iride
        left_dist = eyes_pos[0][0] - iris_pos[0][0]
        right_dist = eyes_pos[1][0] - iris_pos[1][0]

        # se vuoi ridurre i falsi negativi aumentando i falsi positivi
        #if(abs(left_dist)<=self._TRACKING_SENSITIVITY or abs(left_dist)<=self._TRACKING_SENSITIVITY):
        #    return 0

        # se la distanza è minore della sensibilità impostata, ritorniamo 0 (nessun movimento)
        if(abs(left_dist)<=self._TRACKING_SENSITIVITY and abs(left_dist)<=self._TRACKING_SENSITIVITY):
            return 0

        # teniamo solo il segno, che indica la direzione verso cui guardiamo
        left_pos = math.copysign(1, left_dist)
        right_pos = math.copysign(1, right_dist)

        # se vuoi ridurre i falsi negativi aumentando i falsi positivi
        # if left_pos==right_pos==0:
        #    return 0
        # return right_pos

        if left_pos == right_pos:
            return left_pos

        return 0

    def capture(self, detection_enabled, show_rects=False):
        ret, frame = self._video_capture.read()

        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))

            if not detection_enabled:
                return img

            ########################
            ### GAZE DETECTION   ###
            ########################

            # Non facciamo il face detection
            # quindi partiamo dal punto 2

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            rect = dlib.rectangle(self._FACE_RECT[0], self._FACE_RECT[1], self._FACE_RECT[2], self._FACE_RECT[3])
            shape = self._predictor(gray, rect)  # eseguiamo il landmarks detection (2)
            shape = self._shape2np(shape)  # convertiamo in un array numpy

            # (3) è implicito nelle varie operazioni

            mask = self.eyes_on_mask(shape, img.shape[:2], (self._EYE_RIGHT, self._EYE_LEFT), dilate=True) # creiamo la maschera degli occhi (4)

            eyes = cv2.bitwise_and(img, img, mask=mask)  # segmentiamo gli occhi dall'immagine (5)

            # rendiamo tutti i pixel neri bianchi, così solo l'iride rimane nera (6)
            eyes = self.pixels_b2w(eyes)

            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)  # ... però dobbiamo convertire in bw (7)
            thresh = self.thresholding(eyes_gray, self._THRESHOLD_VALUE) # eseguiamo il treesholding per isolare l'iride (8)

            mid = int((shape[42][0] + shape[39][0]) // 2) # calcoliamo il centro orizzontale dell'immagine

            left_cnt = self.contouring(thresh[:, :mid], mid, img, side="L")  # calcoliamo il centro del contorno dell'iride sinistra (9.1)
            right_cnt = self.contouring(thresh[:, mid:], mid, img, side="R") # calcoliamo il centro del contorno dell'iride destra (9.2)

            ########################
            ### GAZE TRACKING    ###
            ########################

            # (1), (2), (3) e (5) sono già stati eseguiti per il gaze detection

            eyes_centers = self._get_eyes_center(shape, (self._EYE_LEFT, self._EYE_RIGHT)) # calcoliamo il centro dell'occhio (4)

            for eye_center in eyes_centers:
                cv2.circle(img, eye_center, 4, (255,0,0),-1)

            for p in zip(self._EYE_LEFT, self._EYE_RIGHT):
                pl, pr = p
                cv2.circle(img, (int(shape[pl][0]), int(shape[pl][1])), 2, (0,255,0), 0)
                cv2.circle(img, (int(shape[pr][0]), int(shape[pr][1])), 2, (0,255,0), 0)

            cv2.circle(img, left_cnt, 3, (0, 0, 255), -1)
            cv2.circle(img, right_cnt, 3, (0, 0, 255), -1)

            if show_rects:
                (x, y, w, h) = self._rect2bb(rect)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


            if self.skip_for>0:
                self.skip_for-=1
                return img
            elif left_cnt==None or right_cnt==None:
                return img

            gaze = self._track_eyes_movements(eyes_centers, (left_cnt, right_cnt)) # tracciamo il movimento (6)

            text = ""

            if gaze==self.GAZE_RIGHT:
                Audio().play("destra.wav")
                text +="DESTRA"
                self.skip_for=self._SKIP_PAUSE
            elif gaze==self.GAZE_LEFT:
                Audio().play("sinistra.wav")
                text +="SINISTRA"
                self.skip_for=self._SKIP_PAUSE

            cv2.putText(img, text, (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        return img