# Number Plate Detection System

An intelligent system for detecting and recognizing vehicle license plates from images and video streams using deep learning.

## Overview

This project implements an end-to-end license plate detection and recognition pipeline that can:
- Detect license plates in images and videos with high accuracy
- Extract and recognize text from detected plates
- Process live video feeds and webcam streams in real-time
- Handle multiple plates in a single frame

## Technology Stack

- **Detection**: YOLOv5 object detection model
- **OCR**: EasyOCR for text recognition
- **Framework**: PyTorch
- **Image Processing**: OpenCV

## Project Structure

```
.
├── scripts/
│   ├── deploy.py       # Main detection and recognition pipeline
│   ├── frame.py        # Video frame extraction utilities
│   └── cars.xml        # Cascade classifier for vehicle detection
├── yolov5/             # YOLOv5 detection framework
├── test_img/           # Sample images for testing
├── data.yaml           # Model configuration
├── Crop_detect.py      # Detection utilities
└── requirements.txt    # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/NITHESH2303/NumberPlateDetection.git
   cd NumberPlateDetection
   ```

2. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare model weights**
   
   The trained model weights are not included due to file size. You'll need to:
   - Train your own model using the instructions below, OR
   - Place pre-trained weights in `weights/best.pt`

## Usage

### Image Detection

Detect license plates in a single image:

```bash
cd scripts
python -c "from deploy import main; main(img_path='../test_img/IMG_2899.JPG')"
```

Press `q` to save and close the output window.

### Video Processing

Process a video file:

```bash
cd scripts
python -c "from deploy import main; main(vid_path='../video.mp4', vid_out='../output/result.mp4')"
```

### Live Webcam

Real-time detection from webcam:

```bash
cd scripts
python -c "from deploy import main; main(vid_path=0, vid_out='../output/live_output.mp4')"
```

## Model Training

To train the detection model on your own dataset:

1. **Prepare your dataset**
   - Organize images and annotations in YOLO format
   - Update `data.yaml` with dataset paths and class information

2. **Configure training parameters**
   ```bash
   cd yolov5
   python train.py --data ../data.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch 16
   ```

3. **Export trained model**
   - After training, copy the best weights:
   ```bash
   cp runs/train/exp/weights/best.pt ../weights/best.pt
   ```

## How It Works

1. **Detection**: The YOLOv5 model identifies license plate regions in the input
2. **Extraction**: Detected regions are cropped and preprocessed
3. **Recognition**: EasyOCR analyzes the cropped regions to extract text
4. **Filtering**: Results are filtered based on confidence thresholds
5. **Output**: Annotated images/videos with detected plates and recognized text

## Configuration

Key parameters can be adjusted in `scripts/deploy.py`:
- `OCR_TH`: OCR confidence threshold (default: 0.2)
- Detection confidence: Minimum 0.55 for bounding boxes
- Frame processing rate for videos

## Output

Processed results are saved to the `output/` directory:
- Images: `output/result_<filename>`
- Videos: `output/<specified_output_name>`

## Performance Notes

- Image processing: Near real-time on CPU
- Video processing: Depends on resolution and hardware
- GPU acceleration recommended for faster inference

## Troubleshooting

**Issue**: Model fails to load
- Ensure `weights/best.pt` exists and is valid
- Check PyTorch installation

**Issue**: Low detection accuracy
- Verify image quality and lighting
- Consider retraining with more diverse data
- Adjust confidence thresholds

**Issue**: OCR errors
- Increase OCR_TH threshold
- Improve image preprocessing

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages:
- torch >= 2.0
- opencv-python
- easyocr
- numpy
- pillow

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is available under the MIT License.
