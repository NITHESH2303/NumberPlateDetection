# Deploying to Hugging Face Spaces

## Prerequisites
- Hugging Face account (free): https://huggingface.co/join
- Your model weights file (`weights/best.pt`)

## Steps

### 1. Create a New Space
1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `license-plate-detection` (or your choice)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) or GPU (faster, paid)

### 2. Upload Files

You need to upload these files to your Space:

```
app.py                  # Main Gradio app
README_HF.md           # Rename to README.md in Space
requirements_hf.txt    # Rename to requirements.txt in Space
weights/best.pt        # Your trained model
yolov5/                # YOLOv5 framework folder
test_img/              # Example images (3-4 samples)
data.yaml              # Model configuration
```

### 3. Upload via Git (Recommended)

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files
cp app.py .
cp requirements_hf.txt requirements.txt
cp README_HF.md README.md
cp -r weights/ .
cp -r yolov5/ .
cp -r test_img/ .
cp data.yaml .

# Commit and push
git add .
git commit -m "feat: add license plate detection app"
git push
```

### 4. Alternative: Upload via Web UI

1. In your Space, click **"Files"** tab
2. Click **"Add file"** > **"Upload files"**
3. Drag and drop all required files
4. Click **"Commit changes"**

### 5. Wait for Build

- Building takes 5-10 minutes
- Watch the build logs in the **"Logs"** tab
- Once complete, your app will be live!

## Important Notes

### Model Weights
- **Size**: Your `weights/best.pt` (40MB) is under the 50GB limit ✅
- If it fails: Use Git LFS for large files
  ```bash
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
  ```

### Hardware Choice
- **CPU Basic (Free)**: Works but slow (10-15 seconds per image)
- **GPU T4 (Paid)**: Fast inference (2-3 seconds per image)
- Start with free CPU, upgrade if needed

### Test Locally First

```bash
pip install gradio
python app.py
```
Open http://localhost:7860 to test

## Troubleshooting

**Build fails with "Out of memory"**
- Remove unused files from yolov5/
- Use CPU hardware instead of GPU during build

**Model not loading**
- Check `weights/best.pt` is uploaded
- Verify path in `app.py` is correct

**Import errors**
- Check all dependencies in `requirements.txt`
- Ensure OpenCV is `opencv-python-headless`

## Your Space URL

After deployment: `https://huggingface.co/spaces/YOUR_USERNAME/license-plate-detection`

Share this link with anyone to let them try your model!
