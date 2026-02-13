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


def process_frame(frame):
    """Process a single frame for detection"""
    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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


def detect_plates(image):
    """Detect license plates in image"""
    if image is None:
        return None, "Please upload an image"
    return process_frame(image)


def detect_plates_video(video_path):
    """Detect license plates in video"""
    if video_path is None:
        return None, "❌ Please upload a video file"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "❌ Error: Could not open video file"
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width == 0 or height == 0 or fps == 0:
            cap.release()
            return None, "❌ Error: Invalid video format"
        
        # Create output video
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} fps")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for speed
            if frame_count % 3 == 0:
                processed_frame, result_text = process_frame(frame)
                out.write(processed_frame)
                if "Detected" in result_text and "plate" in result_text:
                    detection_count += 1
            else:
                out.write(frame)
        
        cap.release()
        out.release()
        
        result_msg = f"✅ Processed {frame_count} frames\n"
        result_msg += f"🎯 Frames with detections: {detection_count}"
        
        return output_path, result_msg
        
    except Exception as e:
        return None, f"❌ Error processing video: {str(e)}"


# Create Gradio interface with tabs
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 License Plate Detection System
    
    Upload an image or video to detect and recognize license plates using YOLOv5 and EasyOCR.
    """)
    
    with gr.Tabs():
        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_button = gr.Button("Detect Plates", variant="primary")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Detection Result")
                    image_text = gr.Textbox(label="Detected Plates", lines=5)
            
            gr.Examples(
                examples=[
                    ["test_img/IMG_2899.JPG"],
                    ["test_img/IMG_2900.JPG"],
                    ["test_img/IMG_2901.JPG"]
                ],
                inputs=image_input
            )
            
            image_button.click(
                fn=detect_plates,
                inputs=image_input,
                outputs=[image_output, image_text]
            )
        
        with gr.Tab("🎥 Video"):
            gr.Markdown("""
            ### Video Processing
            Upload a video file to detect license plates frame-by-frame.
            
            **Note:** Video processing may take 1-2 minutes depending on length.
            """)
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video", sources=["upload"])
                    video_button = gr.Button("🎬 Process Video", variant="primary", size="lg")
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    video_text = gr.Textbox(label="Processing Info", lines=3)
            
            video_button.click(
                fn=detect_plates_video,
                inputs=video_input,
                outputs=[video_output, video_text]
            )
    
    gr.Markdown("""
    ### Features
    - Detects multiple plates in a single frame
    - Recognizes text using OCR
    - Works with various lighting conditions
    
    ### How to use
    1. Choose Image or Video tab
    2. Upload your file
    3. Click the detect/process button
    4. View results with bounding boxes and recognized text
    
    [View source code on GitHub](https://github.com/NITHESH2303/NumberPlateDetection)
    """)

if __name__ == "__main__":
    demo.launch()
