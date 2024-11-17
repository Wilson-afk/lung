from __future__ import division, print_function
from flask import Flask, render_template, request, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from functools import reduce
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchcam.methods import SmoothGradCAMpp

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import os
from flask import send_file
from torchvision import datasets, transforms, models

import io
import base64
import time

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")














app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['DEBUG'] = True
app.secret_key = 'supersecretkey'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']










device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
validator_model = torch.load("saved_models/image_validator_model.pth")
model = torch.load('saved_models/efficientnet_b01_model.pth')


def validate_image(image_path):

    from PIL import Image

    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Modify the final layer for binary classification
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()  # Use Sigmoid for binary classification
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.load_state_dict(validator_model)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).item()
        is_lung_ct = output >= 0.5
    return is_lung_ct





















def predict_image_class_with_smoothgradcam_mc_dropout(image_path, img_size=(224, 224), num_samples=5):
    np.set_printoptions(precision=17, suppress=False)

    def mc_dropout_inference(model, inputs, num_samples=10):
        model.train()
        predictions = []
        for _ in range(num_samples):
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())
        predictions = np.stack(predictions)
        variance = predictions.var(axis=0)
        model.eval()
        return variance

    def enable_mc_dropout(model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'saved_models/efficientnet_b01_model.pth'
    model = torch.load(model_path)
    model.to(device)
    enable_mc_dropout(model)

    target_layer = model.features[-1]
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

    imageq = Image.open(image_path)
    img = imageq.resize(img_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    img_tensor.requires_grad = True

    encoding_map = {'Adenocarcinoma': 0, 'Large Cell Carcinoma': 1, 'Normal': 2, 'Squamous Cell Carcinoma': 3}
    reverse_encoding_map = {v: k for k, v in encoding_map.items()}

    imageq.save('static/outputs/original_image.png')

    all_probs = []
    for _ in range(num_samples):
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = mc_dropout_inference(model, img_tensor, num_samples=10)

    predicted_class = int(np.argmax(mean_probs))
    predicted_class_label = reverse_encoding_map[predicted_class]

    model.eval()
    output = model(img_tensor)
    cam = cam_extractor(predicted_class, output)

    activation_map = cam[0].squeeze().cpu().numpy()
    activation_map_resized = Image.fromarray(activation_map).resize((img_size[0], img_size[1]), Image.BILINEAR)
    activation_map_resized = np.array(activation_map_resized)
    activation_map_resized = (activation_map_resized - activation_map_resized.min()) / (activation_map_resized.max() - activation_map_resized.min())
    heatmap = cm.jet(activation_map_resized)[..., :3]
    overlayed_img = 0.5 * img_array + 0.5 * heatmap

    overlayed_image_pil = Image.fromarray((overlayed_img * 255).astype(np.uint8))
    overlayed_image_pil.save('static/outputs/overlayed_image.png')

    elapsed_time = time.time() - start_time

    def convert_to_fixed(arr, decimals=6):
        return [float(f"{x:.{decimals}f}") for x in arr]

    mean_probs = convert_to_fixed(mean_probs)
    # predicted_class_uncertainty = std_probs[0][predicted_class] * 100

    return predicted_class_label, mean_probs, std_probs[0], elapsed_time




















def get_uncertainty_description(uncertainty_rank):
    """Return a descriptive term for the uncertainty rank."""
    descriptions = {
        1: "smallest",
        2: "second smallest",
        3: "third smallest",
        4: "last"
    }
    return descriptions.get(uncertainty_rank, f"{uncertainty_rank}th smallest")

def get_diagnostic_message(predicted_class_label, max_mean_probs, corresponding_uncertainty, uncertainty_description, confidence_level):
   
    if max_mean_probs > 80 and uncertainty_description == "smallest":
        messages = [
            f"The model predicts {predicted_class_label} with a very high confidence of {max_mean_probs}% and the smallest uncertainty ({corresponding_uncertainty:.2f}%). The prediction is highly reliable, and a review may not be necessary.",
            f"The diagnosis is {predicted_class_label} with an exceptional confidence of {max_mean_probs}% and the lowest uncertainty ({corresponding_uncertainty:.2f}%). This suggests a highly accurate prediction, reducing the need for further review.",
            f"The image shows {predicted_class_label} with an outstanding confidence score of {max_mean_probs}% and the smallest variance in uncertainty ({corresponding_uncertainty:.2f}%). The model's prediction is highly trustworthy, and expert review may not be required."
        ]
        return random.choice(messages)

    if confidence_level == "high":
        messages = [
            f"The image shows {predicted_class_label} with a strong confidence of {max_mean_probs}%, and it has the {uncertainty_description} uncertainty ({corresponding_uncertainty:.2f}%). Clinical review is recommended for confirmation.",
            f"The model confidently predicts {predicted_class_label} with {max_mean_probs}% certainty, and the uncertainty is {uncertainty_description} ({corresponding_uncertainty:.2f}%). Expert review is still advised.",
            f"The diagnosis is {predicted_class_label} with a high confidence score of {max_mean_probs}% and the {uncertainty_description} uncertainty value ({corresponding_uncertainty:.2f}%). Clinical review is suggested to confirm the prediction."
        ]
    elif confidence_level == "medium":
        messages = [
            f"The image suggests {predicted_class_label} with a confidence of {max_mean_probs}% and the {uncertainty_description} uncertainty ({corresponding_uncertainty:.2f}%). Further evaluation is advised.",
            f"The model predicts {predicted_class_label} with a confidence of {max_mean_probs}%. The uncertainty is the {uncertainty_description} ({corresponding_uncertainty:.2f}%), indicating reasonable confidence but suggesting a need for expert validation.",
            f"With a confidence score of {max_mean_probs}% and the {uncertainty_description} uncertainty ({corresponding_uncertainty:.2f}%), the diagnosis leans towards {predicted_class_label}. Clinical validation is necessary."
        ]
    else:  
        messages = [
            f"The diagnosis of {predicted_class_label} comes with a low confidence score of {max_mean_probs}% and the {uncertainty_description} uncertainty ({corresponding_uncertainty:.2f}%). Expert review is strongly recommended.",
            f"The model predicted {predicted_class_label} but with only {max_mean_probs}% confidence and the {uncertainty_description} uncertainty ({corresponding_uncertainty:.2f}%). A detailed review is advised.",
            f"The prediction for {predicted_class_label} has a confidence of {max_mean_probs}% and the {uncertainty_description} uncertainty. Expert evaluation is crucial."
        ]

    selected_message = random.choice(messages)
    return selected_message

def prepare_diagnostic_message(predicted_class_label, mean_probs, std_probs):

    max_prob_index = mean_probs.index(max(mean_probs))

    corresponding_uncertainty = std_probs[max_prob_index]

    max_mean_probs = max(mean_probs)

    sorted_uncertainties = sorted(std_probs)
    uncertainty_rank = sorted_uncertainties.index(corresponding_uncertainty) + 1

    uncertainty_description = get_uncertainty_description(uncertainty_rank)

    session['corresponding_uncertainty'] = corresponding_uncertainty
    session['uncertainty_description'] = uncertainty_description
    session['max_mean_probs'] = max_mean_probs

    if max_mean_probs >= 80:
        confidence_level = "high"
    elif 70 <= max_mean_probs < 80:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    diagnostic_message = get_diagnostic_message(
        predicted_class_label,
        max_mean_probs,
        corresponding_uncertainty,
        uncertainty_description,
        confidence_level
    )

    return diagnostic_message




def generate_pdf_report(predicted_class_label, max_mean_probs, corresponding_uncertainty, uncertainty_description, diagnostic_message):
    report_path = "static/outputs/diagnostic_report.pdf"
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Top Border
    c.setStrokeColor(colors.darkblue)
    c.setLineWidth(2)
    c.line(0.5 * inch, height - 0.5 * inch, width - 0.5 * inch, height - 0.5 * inch)

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(width / 2, height - 1 * inch, "📝 Diagnostic Report")

    # Disclaimer below the title
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.red)
    c.drawCentredString(width / 2, height - 1.3 * inch, "This diagnosis has not been reviewed by a medical practitioner.")

    # Date Section
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.red)
    date_str = datetime.now().strftime("%B %d, %Y")
    c.drawString(1 * inch, height - 1.7 * inch, f"📅 Date of Diagnosis: {date_str}")

    # Separator Line
    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(1)
    c.line(0.5 * inch, height - 2 * inch, width - 0.5 * inch, height - 2 * inch)

    # Display Original and Grad-CAM Images
    try:
        original_image = ImageReader("static/outputs/original_image.png")
        gradcam_image = ImageReader("static/outputs/overlayed_image.png")
        c.drawImage(original_image, 1 * inch, height - 5.5 * inch, width=3 * inch, height=3 * inch)
        c.drawImage(gradcam_image, 4.5 * inch, height - 5.5 * inch, width=3 * inch, height=3 * inch)
        c.setFont("Helvetica", 10)
        c.drawString(1.5 * inch, height - 5.7 * inch, "Original Image")
        c.drawString(5 * inch, height - 5.7 * inch, "Grad-CAM Visualization")
    except Exception as e:
        print(f"Error loading images: {e}")

    # Diagnostic Summary Section
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkblue)
    c.drawString(1 * inch, height - 6.5 * inch, "🧬 Diagnostic Summary:")

    # Diagnostic Details
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(1 * inch, height - 7.0 * inch, f"• Diagnosis: Lung {predicted_class_label}")
    c.drawString(1 * inch, height - 7.5 * inch, f"• Model Confidence: {max_mean_probs:.0f}% (Moderate Confidence)")
    c.drawString(1 * inch, height - 8.0 * inch, f"• Uncertainty Level: {uncertainty_description.capitalize()}")

    # Separator Line
    c.line(0.5 * inch, height - 8.3 * inch, width - 0.5 * inch, height - 8.3 * inch)

    # Interpretation Section
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.orange)
    c.drawString(1 * inch, height - 8.8 * inch, "💡 Interpretation:")

    # Interpretation Text
    c.setFont("Helvetica", 12)
    interpretation_text = (
        f"The AI model predicts lung {predicted_class_label} with a confidence of {max_mean_probs:.0f}%. "
        f"The uncertainty level is considered {uncertainty_description}, indicating consistent internal prediction. "
        "However, a review by a healthcare expert is recommended to confirm the diagnosis."
    )
    text_object = c.beginText(1 * inch, height - 9.5 * inch)
    text_object.setFont("Helvetica", 12)
    text_object.setFillColor(colors.black)
    wrapped_text = "\n".join([interpretation_text[i:i+80] for i in range(0, len(interpretation_text), 80)])
    text_object.textLines(wrapped_text)
    c.drawText(text_object)

    # Bottom Border
    c.setStrokeColor(colors.darkblue)
    c.setLineWidth(2)
    c.line(0.5 * inch, 0.75 * inch, width - 0.5 * inch, 0.75 * inch)

    # Footer Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    disclaimer_text = (
        "Disclaimer: This report is generated by an AI diagnostic system and should not be used as a substitute "
        "for professional medical advice. Please consult a healthcare expert for a thorough evaluation."
    )
    footer_text = c.beginText(0.5 * inch, 0.5 * inch)
    footer_text.setFont("Helvetica-Oblique", 10)
    footer_text.setFillColor(colors.grey)
    wrapped_disclaimer = "\n".join([disclaimer_text[i:i+100] for i in range(0, len(disclaimer_text), 100)])
    footer_text.textLines(wrapped_disclaimer)
    c.drawText(footer_text)

    # Save the PDF
    c.save()

    return report_path











@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


@app.route('/feedback', methods=["GET", "POST"])
def feedback():
    return render_template('feedback.html')






@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'csvfile' not in request.files:
            flash('No file part provided. Please upload an image.', 'error')
            return redirect(request.url)
        
        file = request.files['csvfile']
        if file.filename == '':
            flash('No file selected. Please choose an image to upload.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            if not os.path.isdir(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Validate the image using the image validator model
            is_valid_image = validate_image(filepath)
            if not is_valid_image:
                flash('The uploaded image is not a valid lung CT scan. Please upload a proper lung CT scan image.', 'error')
                return redirect(request.url)

            # flash('File uploaded successfully. Processing your image...', 'success')
            # return redirect(url_for('main'))

            predicted_class_label, mean_probs, std_probs, elapsed_time = \
                predict_image_class_with_smoothgradcam_mc_dropout(filepath)
            
            std_probs = std_probs.tolist()
            mean_probs = [int(np.ceil(prob * 100)) for prob in mean_probs] 
            std_probs = [round(std * 100, 2) for std in std_probs]
            show_submit_button = max(mean_probs) < 90
            diagnostic_message = prepare_diagnostic_message(predicted_class_label, mean_probs, std_probs)

            session['predicted_class_label'] = predicted_class_label
            session['mean_probs'] = mean_probs
            session['max_mean_probs'] = max(mean_probs)
            session['show_submit_button'] = show_submit_button
            session['std_probs'] = std_probs
            session['elapsed_time'] = elapsed_time
            session['diagnostic_message'] = diagnostic_message

            return redirect(url_for('main'))

        else:
            flash('Only image files (png, jpg, jpeg, gif) are allowed.', 'error')
            return redirect(request.url)

    return render_template('upload.html')





@app.route('/main', methods=["GET", "POST"])
def main():
    predicted_class_label = session.get('predicted_class_label')
    mean_probs = session.get('mean_probs')
    max_mean_probs = session.get('max_mean_probs')
    show_submit_button = session.get('show_submit_button')
    std_probs = session.get('std_probs')
    diagnostic_message = session.get('diagnostic_message')
    elapsed_time = session.get('elapsed_time')

    return render_template(
        'main.html',
        predicted_class_label=predicted_class_label,
        mean_probs=mean_probs,
        max_mean_probs=max_mean_probs,
        show_submit_button=show_submit_button,
        std_probs=std_probs,
        diagnostic_message=diagnostic_message,
        elapsed_time=elapsed_time
    )


@app.route('/download_report', methods=["GET"])
def download_report():
    predicted_class_label = session.get('predicted_class_label')
    max_mean_probs = session.get('max_mean_probs')
    corresponding_uncertainty = session.get('corresponding_uncertainty')
    uncertainty_description = session.get('uncertainty_description')
    diagnostic_message = session.get('diagnostic_message')

    report_path = generate_pdf_report(
        predicted_class_label,
        max_mean_probs,
        corresponding_uncertainty,
        uncertainty_description,
        diagnostic_message
    )

    return send_file(report_path, as_attachment=True)



@app.route('/review', methods=["GET", "POST"])
def review():
    predicted_class_label = session.get('predicted_class_label')
    mean_probs = session.get('mean_probs')
    max_mean_probs = session.get('max_mean_probs')
    show_submit_button = session.get('show_submit_button')
    std_probs = session.get('std_probs')
    diagnostic_message = session.get('diagnostic_message')
    elapsed_time = session.get('elapsed_time')


    return render_template(
        'review.html',
        predicted_class_label=predicted_class_label,
        mean_probs=mean_probs,
        max_mean_probs=max_mean_probs,
        show_submit_button=show_submit_button,
        std_probs=std_probs,
        diagnostic_message=diagnostic_message,
        elapsed_time=elapsed_time
    )


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')



























