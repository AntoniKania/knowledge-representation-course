import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained neural network
model = tf.keras.models.load_model("number_predictions_convolutional_2.keras")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresholded

# Function to extract the table from the image
def extract_table(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv2.contourArea)  # Assuming the table is the largest contour
    x, y, w, h = cv2.boundingRect(table_contour)
    table = image[y:y + h, x:x + w]
    return table

# Function to segment entries in the table
def segment_cells(table_image):
    cells = []
    num_rows, num_cols = 3, 3  # Assuming a 3x3 grid
    cell_h, cell_w = table_image.shape[0] // num_rows, table_image.shape[1] // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            cell = table_image[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cells.append(cell)
    return cells

# Function to predict digits from cells
def predict_digits(cells):
    phone_number = ""

    for cell in cells:
        resized = cv2.resize(cell, (28, 28))  # Resize to the input size expected by the model
        normalized = resized / 255.0
        input_data = normalized.reshape(1, 28, 28, 1)
        prediction = model.predict(input_data)
        digit = np.argmax(prediction)
        phone_number += str(digit)

    return phone_number

# Main processing pipeline
def process_document(image_path):
    processed_image = preprocess_image(image_path)
    table_image = extract_table(processed_image)
    cells = segment_cells(table_image)
    phone_number = predict_digits(cells)
    return phone_number

# Example usage
if __name__ == "__main__":
    image_path = "form_with_phone_number.png"
    phone_number = process_document(image_path)
    print(f"Extracted Phone Number: {phone_number}")