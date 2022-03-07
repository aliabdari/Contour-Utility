import imutils
import cv2
import numpy as np
from scipy.spatial import distance

from tensorflow.keras.models import load_model


def do_detect_circle(image, frame_number):
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 20, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    mask = np.zeros(image.shape[: 2], np.uint8)
    cv2.drawContours(mask, c, -1, 255, -1)
    mean = cv2.mean(image, mask=mask)

    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * 1)
    cY = int((M["m01"] / M["m00"]) * 1)

    return cX, cY, (w // 2 + h // 2) // 2, (int(mean[0]), int(mean[1]), int(mean[2]))


def find_permitted_numbers(image, x_center, y_center, large_circle_radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # dilated = cv2.dilate(thresh.copy(), None, iterations=1)
    kernelSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    thresh = cv2.threshold(opening, 3, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    contours = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    permitted_contours = []

    for c in contours:

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]) * 1)
        cY = int((M["m01"] / M["m00"]) * 1)
        dist = distance.euclidean((cX, cY), (x_center, y_center))
        # print("dist", dist)
        if dist < large_circle_radius:
            permitted_contours.append(c)

    print("number of permitted figures =", len(permitted_contours))
    return permitted_contours


def get_resized_image(thresh):
    shape_thresh = thresh.shape
    bigger = False
    if shape_thresh[0] > 28 or shape_thresh[1] > 28:
        bigger = True
    if not bigger:
        x_pad_left = (28 - shape_thresh[0]) // 2
        x_pad_right = 28 - shape_thresh[0] - x_pad_left
        y_pad_left = (28 - shape_thresh[1]) // 2
        y_pad_right = 28 - shape_thresh[1] - y_pad_left
        res = np.pad(thresh, ((x_pad_left, x_pad_right), (y_pad_left, y_pad_right)), 'edge')
    else:
        res = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_LANCZOS4)
    return res


def detect_the_numbers(contours, image, model_file, frame_number=None):
    model = load_model(model_file)
    numbers = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print("wh", w * h)
        if 1500 > w * h > 200:
            cropped_image = image[y-1:y + h+1, x-1:x + w+1]
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 5, 240, cv2.THRESH_BINARY)[1]
            resized_image = get_resized_image(thresh)
            output = cv2.connectedComponentsWithStats(
                thresh, 4, cv2.CV_32S)
            # resized_image = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            # dilated = cv2.dilate(resized_image.copy(), None, iterations=0.2)
            expand_dim = np.expand_dims(resized_image, axis=(0, -1))
            result = model.predict(expand_dim)
            # print(result)
            numbers.append(np.argmax(result))
    return numbers
