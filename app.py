import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import cv2
import pytesseract
import base64
from pytesseract import Output


app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get the uploaded file
        file = request.files.get('image')
        option = request.form.get('budget', '')

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save the file to a directory
        os.makedirs('static', exist_ok=True)
        filename = os.path.join('static', file.filename)
        file.save(filename)

        # Read the file from the saved location
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # Check if the image is valid
        if image is None:
            return jsonify({"error": "Invalid image format or corrupted file"}), 400

        # Perform OCR
        result = read_img(image)

        if not result.strip():
            textDetected = False
            print("Empty String")
            result = "No text detected"
            to_send = ''
        else:
            textDetected = True
            print("Text detected")

            # Perform text detection and annotate the image
            d = pytesseract.image_to_data(image, output_type=Output.DICT)
            image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            n_boxes = len(d['text'])

            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert the annotated image to base64
            _, image_content = cv2.imencode('.jpg', image)
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            to_send = f'data:image/jpg;base64, {encoded_image}'

        return render_template('index.html', textDetected=textDetected, prediction_text=result, image_to_show=to_send, init=True)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# Dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# Erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# Read Image
def read_img(img):
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding for better contrast
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Remove noise
        processed_img = cv2.medianBlur(binary, 3)

        # Perform OCR with specified language (e.g., English)
        text = pytesseract.image_to_string(processed_img, lang='eng+hin+mar')

        return text

    except Exception as e:
        print(f"Error in OCR: {e}")
        return ""


if __name__ == "__main__":
    app.run(debug=True, port=5000)


