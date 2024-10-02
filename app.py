import matplotlib
matplotlib.use('Agg')

import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import random
import ssl
import cv2
from flask import Flask, render_template, request, jsonify
import base64
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Constants
CONFIDENCE_THRESHOLD = 0.99
IMAGE_SIZE = 224
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Prediction history
prediction_history = []
    
# function for Loading the model (.pth)
def load_model(device):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASSES))
    # model path
    model.load_state_dict(torch.load('./fine_tuned_efficientnet_v2_s.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = load_model(device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# function for prediction of image if there is tumor or not
def predict_image(image, model, device, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        max_prob, predicted_class = torch.max(probs, 1)
        
    confidence = max_prob.item()
    probabilities = probs[0].cpu().detach().numpy()
    
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction = CLASSES[predicted_class.item()]
    else:
        prediction = "no_tumor" if confidence >= 0.5 else "not_valid_mri"
    
    return prediction, confidence, probabilities

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def is_brain_mri(image):
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = auto_canny(blurred)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contoured_image = image_cv.copy()
    if len(contours) > 0:
        cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
        return True, contoured_image
    else:
        return False, contoured_image


# / route for rendering index.html page
@app.route('/')
def abcd():
    return render_template("index.html");

# /predict API for prediction of brain tumor
@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # if file is not present in frontend return error
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        # if no file selected then show message
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # if file is present then process the image and show results
        if file:
            image = Image.open(file.stream).convert('RGB')
            is_mri, contoured_image = is_brain_mri(image)
            
            prediction, confidence, probabilities = predict_image(image, model, device, transform)
            
            # Log prediction in history
            prediction_history.append({
                'filename': file.filename,
                'prediction': prediction,
                'confidence': confidence
            })
            
            # Convert images to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            original_image = base64.b64encode(buffered.getvalue()).decode()
            
            buffered = io.BytesIO()
            Image.fromarray(contoured_image).save(buffered, format="PNG")
            contoured_image = base64.b64encode(buffered.getvalue()).decode()
            
            plt.figure(figsize=(10, 5))
            plt.bar(CLASSES, probabilities)
            plt.title('Prediction Probabilities')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            
            buffered = io.BytesIO()
            plt.savefig(buffered, format="PNG")
            chart_image = base64.b64encode(buffered.getvalue()).decode()
            plt.close()  

            # if arguments is ?format=json then show json response
            if request.args.get('format') == 'json':
                return jsonify({
                    'original_image': original_image,
                    'contoured_image': contoured_image,
                    'is_mri': is_mri,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist() 
                })

            # If not requesting JSON, render HTML
            return render_template('result.html', 
                                   original_image=original_image,
                                   contoured_image=contoured_image,
                                   is_mri=is_mri,
                                   prediction=prediction,
                                   confidence=confidence,
                                   chart_image=chart_image)
    return render_template('index.html')


### Model Information API
@app.route('/model_info', methods=['GET'])
def model_info():
    model_summary = []
    model_info_str = f"Model Architecture: EfficientNet-V2-S\nNumber of Classes: {len(CLASSES)}\nClasses: {CLASSES}"
    return model_info_str

### Prediction History API
@app.route('/prediction_history', methods=['GET'])
def get_prediction_history():
    return jsonify({"history": prediction_history})


# Quiz questions and answers
QUESTIONS = [
    {"question": "Do you experience frequent headaches?", "options": ["Yes", "No", "Sometimes"]},
    {"question": "Do you feel nausea or vomiting often?", "options": ["Yes", "No", "Rarely"]},
    {"question": "Have you had seizures?", "options": ["Yes", "No", "Occasionally"]},
    {"question": "Do you experience vision problems?", "options": ["Yes", "No", "Occasionally"]},
    {"question": "Do you feel numbness or weakness in your limbs?", "options": ["Yes", "No", "Sometimes"]},
    {"question": "Do you have trouble with balance?", "options": ["Yes", "No", "Occasionally"]},
    {"question": "Have you experienced changes in personality or memory?", "options": ["Yes", "No", "Sometimes"]},
    {"question": "Do you have trouble speaking or understanding?", "options": ["Yes", "No", "Sometimes"]},
    {"question": "Do you experience hearing loss?", "options": ["Yes", "No", "Occasionally"]},
    {"question": "Do you often feel dizzy or lightheaded?", "options": ["Yes", "No", "Sometimes"]},
]

@app.route('/quiz', methods=['GET'])
def get_quiz():
    # Serves a random set of 5 questions
    selected_questions = random.sample(QUESTIONS, 5)
    return jsonify({"questions": selected_questions})

# API for showing results of quiz
@app.route('/quiz_result', methods=['POST'])
def submit_quiz():
    data = request.json
    answers = data.get("answers", [])
    
    # detects answer
    yes_count = answers.count("Yes")
    
    if yes_count >= 4:
        result = "Severe Brain Tumor Symptoms"
    elif 2 <= yes_count < 4:
        result = "Mild Brain Tumor Symptoms"
    else:
        result = "No Brain Tumor Symptoms"

    return jsonify({"result": result})

# rendering quiz.html page
@app.route('/quiz_page')
def quiz_page():
    return render_template('quiz.html');

if __name__ == '__main__':
    import threading
    app.run(debug=True, threaded=True)
