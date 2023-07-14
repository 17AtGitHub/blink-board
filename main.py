import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def midpoint(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)


font = cv2.FONT_HERSHEY_PLAIN


def get_blinking_ratio(st, landmarks):
    if landmarks:
        left_point = (landmarks.part(st).x, landmarks.part(st).y)
        right_point = (landmarks.part(st+3).x, landmarks.part(st+3).y)
        center_top_left = landmarks.part(st+1)
        center_top_right = landmarks.part(st+2)
        center_top = midpoint(center_top_left, center_top_right)
        center_bottom = midpoint(landmarks.part(st+5), landmarks.part(st+4))

        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        ver_line_len = hypot(
            center_top[0]-center_bottom[0], center_top[1]-center_bottom[1])
        hor_line_len = hypot(
            left_point[0]-right_point[0], left_point[1]-right_point[1])

        ratio = hor_line_len/ver_line_len
        # print(hor_line_len/ver_line_len) ## better to use the ratio as with distance, the vertical line length can decrease/increase so setting an absolute threshold is not a good idea
        return ratio


def get_gaze_ratio(st, landmarks):

    eye_region = np.array([(landmarks.part(st).x, landmarks.part(st).y),
                           (landmarks.part(st+1).x, landmarks.part(st+1).y),
                           (landmarks.part(st+2).x, landmarks.part(st+2).y),
                           (landmarks.part(st+3).x, landmarks.part(st+3).y),
                           (landmarks.part(st+4).x, landmarks.part(st+4).y),
                           (landmarks.part(st+5).x, landmarks.part(st+5).y),
                           ], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    # cv2.imshow("mask", mask)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_eye = cv2.bitwise_and(gray, gray, mask=mask)
    # cv2.imshow("eye_region", gray_eye)

    # to get a frame around the eye region, from the landmark points:
    # left most point, is the one with the minx:
    min_x = np.min(eye_region[:, 0])
    # right most point for the eye frame is the one with the max x coordinate
    max_x = np.max(eye_region[:, 0])
    # # top-most pint for the eye frame, is the one with the min y coordinate
    min_y = np.min(eye_region[:, 1])
    # # bottom-most pnt for the frame is the one with max y
    max_y = np.max(eye_region[:, 1])
    # print(min_x)
    # print(min_y)
    only_eye = gray_eye[min_y:max_y, min_x:max_x]
    # pixels greater than 70 grayscale get a value 255: white
    _, threshold_eye = cv2.threshold(only_eye, 70, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', threshold_eye)
    # # in order to distinguish left and right gaze, i need to separate the left and right eye regions
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0: int(width/2)]
    left_side_white = np.count_nonzero(left_side_threshold)
    right_side_threshold = threshold_eye[0: max_y, int(width/2): width]
    right_side_white = np.count_nonzero(right_side_threshold)
    # based on the differential black and white cells in each frame, left and right gaze can be determined
    if left_side_white > 0 and right_side_white > 0:  # to avoid div by 0 err
        gaze_ratio = left_side_white/right_side_white
        return gaze_ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        landmarks = predictor(gray, face)

        # BLINKING_DETECTION
        left_eye_ratio = get_blinking_ratio(36, landmarks)
        right_eye_ratio = get_blinking_ratio(42, landmarks)
        if left_eye_ratio and right_eye_ratio:
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if (blinking_ratio > 5.5):
                cv2.putText(frame, 'BLINKING', (50, 150),
                            font, 3, (255, 0, 0), 2)

        # GAZE_DETECTION
        left_eye_gaze_ratio = get_gaze_ratio(36, landmarks)
        right_eye_gaze_ratio = get_gaze_ratio(42, landmarks)
        indicator = np.zeros((500, 500, 3), np.uint8)
        if left_eye_gaze_ratio and right_eye_gaze_ratio:
            gaze_ratio = (left_eye_gaze_ratio + right_eye_gaze_ratio)/2
            # cv2.putText(frame, str(gaze_ratio), (50, 300),
            #             font, 2, (0, 0, 255), 3)
            if gaze_ratio <= 0.8:
                indicator[:] = (0, 0, 255)
                cv2.putText(frame, 'RIGHT', (50, 200), font, 2, (0, 0, 255), 3)
            elif gaze_ratio <= 1.3:
                indicator[:] = (0, 255, 0)
                cv2.putText(frame, 'CENTER', (50, 200),
                            font, 2, (0, 0, 255), 3)
            else:
                indicator[:] = (255, 0, 0)
                cv2.putText(frame, 'LEFT', (50, 200), font, 2, (0, 0, 255), 3)

        cv2.imshow('indicator', indicator)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
