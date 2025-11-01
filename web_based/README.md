# SwinHAFNet & UNet Streamlit Demo

Interactive web interface for brain tumor segmentation using SwinHAFNet or UNet models.

## Features

- Upload up to 5 medical images (PNG/JPG)
- Choose between UNet and SwinHAFNet models
- Adjustable segmentation threshold
- CPU/GPU inference support
- Real-time visualization with overlay masks

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### 2. Prepare Model Checkpoints

Place your trained model checkpoints in the `checkpoints/` directory:
- `checkpoints/unet_model.pth` - UNet model
- `checkpoints/swin_hafnet.pth` - SwinHAFNet model

### 3. Run the Application

```bash
# From project root directory
streamlit run web_based/app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select Model**: Choose between UNet or SwinHAFNet in the sidebar
2. **Select Device**: Choose CPU or CUDA (if available)
3. **Upload Images**: Click "Browse files" to upload 1-5 images. Or check the "Load 3 sample images" box to use 3 sample images.
4. **Adjust Threshold**: Use the slider to change mask threshold (default: 0.5)
5. **Run Inference**: Click "Run inference" button
6. **View Results**: Segmentation masks appear with red overlay on tumors

## Model Support

### SwinHAFNet
- Hierarchical Attention Fusion architecture
- Swin Transformer backbone
- Input: 512×512 RGB images
- Output: Binary segmentation (tumor/background)

### UNet
- EfficientNet-B0 encoder
- Standard UNet decoder
- Input: 512×512 RGB images
- Output: Binary segmentation (tumor/background)

## Technical Details

- **Preprocessing**: Images resized to 512×512
- **Inference**: Softmax probabilities on 2 classes (background, tumor)
- **Threshold**: Adjustable threshold for binary mask generation
- **Output Format**: Red overlay on original image showing detected tumor regions

## Notes

- 512x512 input images is better for models
- GPU inference recommended for faster processing
- Checkpoint loading handles multiple state dict formats
- If no checkpoint found, models use random weights (for testing only)

## Troubleshooting

**Import errors**: Ensure you run from project root and all dependencies are installed

**CUDA errors**: Check CUDA availability with `torch.cuda.is_available()`

**Model loading fails**: Verify checkpoint paths and format compatibility

**Low accuracy**: Ensure you're using trained checkpoints, which satisfy SWIN-HAFNet (or U-Net) architecture, not random weights
