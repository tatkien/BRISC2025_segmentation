import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from web_based.model_utils import load_model, preprocess_image, predict_masks
import streamlit_paste_button

st.set_page_config(page_title="Brain Tumor Image Segmentation", layout="wide")

st.title("Brain Tumor Image Segmentation Demo")
st.markdown("Upload up to 5 images (PNG/JPG) and click `Run` to get segmentation masks.")

st.markdown("Note: In web deployment, no cuda supported")
# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_path = st.selectbox("Model", options=["UNet", "SwinHAFNet"])
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5)
    run_button = st.button("Run inference")

# File uploader (allow multiple)
uploaded_files = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
if uploaded_files is None:
    uploaded_files = []

if len(uploaded_files) > 5:
    st.warning("Please upload at most 5 images.")
    uploaded_files = uploaded_files[:5]

# Display uploaded images and placeholders for outputs
cols = st.columns(min(5, max(1, len(uploaded_files))))
images = []
for i, up in enumerate(uploaded_files):
    try:
        image = Image.open(io.BytesIO(up.read())).convert("RGB")
        images.append(image)
        cols[i].image(image, width='stretch', caption=up.name)
    except Exception as e:
        st.error(f"Failed to read {up.name}: {e}")

model = None
if run_button and len(images) > 0:
    with st.spinner("Loading model and running inference..."):
        # Determine model path and type
        if model_path == "SwinHAFNet":
            checkpoint_path = "checkpoints/swin_hafnet_model.pth"
            model_type = "swin"
        else:  # UNet
            checkpoint_path = "checkpoints/unet_model.pth"
            model_type = "unet"
        
        print(f"Loading {model_path} from {checkpoint_path}")
        model = load_model(model_path=checkpoint_path, device=device, model_type=model_type)
        results = []
        for img in images:
            inp = preprocess_image(img, device=device)
            probs = predict_masks(model, inp)  # (1, 2, H, W)
            mask = probs[:, 1:2, :, :].cpu().numpy()[0, 0]
            results.append((img, mask))

    # Show results side-by-side
    res_cols = st.columns(len(results))
    for i, (img, mask) in enumerate(results):
        overlay = (mask > threshold).astype(np.uint8) * 255
        overlay_img = Image.fromarray(overlay).convert("L").resize(img.size)
        # make an RGBA overlay
        rgba = Image.new("RGBA", img.size)
        rgba.paste(img.convert("RGBA"))
        red = Image.new("RGBA", img.size, (255,0,0,120))
        rgba.paste(red, mask=overlay_img)
        res_cols[i].image(rgba, width='stretch', caption=f"Result {i+1}")

elif run_button and len(images) == 0:
    st.warning("No images uploaded.")

st.markdown("---")