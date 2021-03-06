import cv2
import numpy as np

from helpers import calculate_crop


def face_tracker():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video = cv2.VideoCapture(0)
    image = cv2.imread('stream.jpg')
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    last_working_crop_start = 0
    last_working_crop_end = width-1

    while True:
        check, color_frame = video.read()
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (x, y, w, h) in faces:
            crop_start, crop_end = calculate_crop(x)
            if crop_start > 0: last_working_crop_start = crop_start
            if crop_end < width: last_working_crop_end = crop_end

            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        crop = image[0:400, last_working_crop_start:last_working_crop_end]

        cv2.imshow('Live', color_frame)
        cv2.imshow('Image', crop)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cleanup(video)


def motion_tracker():
    first_frame = None
    video = cv2.VideoCapture(0)

    while True:
        detected = False
        check, color_frame = video.read()
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            continue

        delta_frame = cv2.absdiff(first_frame, gray)
        thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame.copy(), None, iterations=3)

        (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 20000:
                detected = False
                continue
            detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        show_windows(delta_frame, thresh_frame, color_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        cleanup(video)


def show_windows(delta_frame, thresh_frame, color_frame):
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color", color_frame)


def cleanup(video):
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # motion_tracker()
    face_tracker()
