import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def find_largest_rectangle(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    best_rectangle = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            if area > max_area:
                best_rectangle = box
                max_area = area

    return best_rectangle


def order_rectangle_points(pts):
    rect_pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top_pts = sorted(rect_pts[:2], key=lambda p: p[0])
    bottom_pts = sorted(rect_pts[2:], key=lambda p: p[0])

    return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype="float32")


def warp_perspective(image, src_pts):
    w = int(np.linalg.norm(src_pts[1] - src_pts[0]))
    h = int(np.linalg.norm(src_pts[2] - src_pts[1]))

    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return cv2.warpPerspective(image, M, (w, h)), w


def extract_digits(warped_image, w):
    digit_width = w // 9
    digits = []

    for i in range(9):
        digit_x = i * digit_width

        digit_crop = warped_image[:, digit_x:digit_x + digit_width]

        digit = cv2.resize(digit_crop, (28, 28), interpolation=cv2.INTER_LINEAR)
        digit = digit.astype("float32") / 255.0
        input_arr = np.reshape(digit, (1, 28, 28))

        digits.append(input_arr)

    return digits


def predict_digit(model, image):
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return str(predicted_digit)


if __name__ == '__main__':
    image_path = "./form_with_phone_number_3.png"
    image = cv2.imread(image_path)

    model_filename = "number_predictions_convolutional_2.keras"
    model = load_model(model_filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    number_input_box = find_largest_rectangle(thresh)

    ordered_pts = order_rectangle_points(number_input_box)
    aligned_image, w = warp_perspective(thresh, ordered_pts)

    if aligned_image is not None:
        digit_images = extract_digits(aligned_image, w)
        predicted_digits = [predict_digit(model, digit) for digit in digit_images]
        phone_number = "".join(predicted_digits)
        print("Phone Number:", phone_number)

    else:
        print("Perspective warping failed.")

