"""
Gradio Web Interface for License Plate Detection
Author: Nithesh Kanna
"""
import gradio as gr
import torch
import cv2
import numpy as np
import easyocr
from pathlib import Path

# Initialize OCR
EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

# Load model
print("Loading YOLOv5 model...")
model = torch.hub.load('yolov5', 'custom', source='local', path='weights/best.pt', force_reload=False)
print("Model loaded successfully!")


def recognize_plate_easyocr(img, coords, reader, region_threshold):
    """Extract text from detected plate region using EasyOCR"""
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    if nplate.size == 0:
        return []
    
    ocr_result = reader.readtext(nplate)
    
    rectangle_size = nplate.shape[0] * nplate.shape[1]
    plate = []
    
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    
    return plate


def detect_plates(image):
    """Detect license plates and recognize text"""
    if image is None:
        return None, "Please upload an image"
    
    # Convert to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(img)
    
    # Get predictions
    predictions = results.xyxyn[0]
    labels = predictions[:, -1]
    coords = predictions[:, :-1]
    
    detected_plates = []
    
    # Process each detection
    for i, row in enumerate(coords):
        if row[4] >= 0.55:  # Confidence threshold
            x_shape, y_shape = img.shape[1], img.shape[0]
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            
            # Recognize plate text
            plate_coords = [x1, y1, x2, y2]
            plate_text = recognize_plate_easyocr(
                img=img, 
                coords=plate_coords, 
                reader=EASY_OCR, 
                region_threshold=OCR_TH
            )
            
            if len(plate_text) == 1:
                plate_text = plate_text[0].upper()
            else:
                plate_text = ' '.join(plate_text)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(img, str(plate_text), (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detected_plates.append({
                'text': plate_text,
                'confidence': float(row[4])
            })
    
    # Convert back to BGR for display
    output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create result text
    if detected_plates:
        result_text = f"✅ Detected {len(detected_plates)} plate(s):\n\n"
        for i, plate in enumerate(detected_plates, 1):
            result_text += f"{i}. Text: {plate['text']} (Confidence: {plate['confidence']:.2%})\n"
    else:
        result_text = "❌ No license plates detected"
    
    return output_img, result_text


# Create Gradio interface
demo = gr.Interface(
    fn=detect_plates,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Detection Result"),
        gr.Textbox(label="Detected Plates", lines=5)
    ],
    title="🚗 License Plate Detection System",
    description="""
    Upload an image to detect and recognize license plates.
    
    **Features:**
    - Detects multiple plates in a single image
    - Recognizes text using OCR
    - Works with various lighting conditions
    
    **How to use:**
    1. Upload an image containing vehicles with license plates
    2. Wait for processing (takes 5-10 seconds)
    3. View detected plates with bounding boxes and recognized text
    """,
    examples=[
        ["test_img/IMG_2899.JPG"],
        ["test_img/IMG_2900.JPG"],
        ["test_img/IMG_2901.JPG"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
