import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, abort
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from ultralytics import YOLO

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'supersecretkey'

# Parameters
img_height, img_width = 224, 224
threshold = 0.9  # Similarity threshold

# Load the MobileNetV2 model with the saved weights
base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(img_height, img_width, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
])

# Build the model by running a dummy pass
model.build((None, img_height, img_width, 3))

# Now load the weights
model.load_weights('custom_mobilenet_weights.weights.h5')

# Load YOLO model
model1_path = "best40.pt"
model1 = YOLO(model1_path, task="detect")

# Define a function to preprocess and extract features from an image
def extract_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(img_height, img_width))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features

# Load dataset and extract features
def load_dataset_and_extract_features(dataset_path, model):
    features = []
    img_paths = []
    class_names = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('jpeg', 'jpg', 'png')):
                img_path = os.path.join(root, file)
                feature = extract_features(img_path, model)
                features.append(feature)
                img_paths.append(img_path)
                class_name = os.path.basename(root)
                class_names.append(class_name)
    return features, img_paths, class_names

# Define the dataset path
dataset_path = 'uploads'

# Extract features and class names from the dataset
existing_features, img_paths, class_names = load_dataset_and_extract_features(dataset_path, model)

# Function to check if an image belongs to an existing class
def check_image(new_features, existing_features, class_names, threshold=0.9):
    similarities = [cosine_similarity(new_features, existing)[0][0] for existing in existing_features]
    
    max_similarity = max(similarities)
    max_index = similarities.index(max_similarity)
    
    if max_similarity > threshold:
        return True, max_similarity, class_names[max_index]
    else:
        return False, max_similarity, None

@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        # Check if the file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
        
        # Open the image and convert to numpy array for YOLO
        image = Image.open(temp_file.name)
        image_np = np.array(image)

        # Perform inference with YOLO
        results = model1(image_np, conf=0.5, imgsz=640, iou=0.0)

        # Process YOLO results
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes.xyxy.tolist()

            if boxes:
                box = boxes[0]  # Use the first detected box
                x1, y1, x2, y2 = map(int, box[:4])

                # Crop the image based on the first bounding box
                cropped_image = image.crop((x1, y1, x2, y2))

                # Save the cropped image to a temporary file
                cropped_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_image.jpg')
                cropped_image.save(cropped_image_path, format="JPEG")

                # Extract features from the cropped image
                new_features = extract_features(cropped_image_path, model)
                matched, similarity, predicted_class = check_image(new_features, existing_features, class_names, threshold)

                result = {
                    'matched': matched,
                    'similarity': similarity,
                    'predicted_class': predicted_class
                }

                return jsonify(result), 200
            else:
                return jsonify({"error": "No objects detected in the image."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
