import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
import pandas as pd
from web_based.model_utils import load_model, preprocess_image, predict_masks


def load_default_images():
    """Load 3 random images from test dataset"""
    try:
        test_csv_path = os.path.join(os.getcwd(), 'dataset', 'test.csv')
        if os.path.exists(test_csv_path):
            print("Loading default images from:", test_csv_path)
            df_test = pd.read_csv(test_csv_path)
            # Sample 3 random images
            sampled = df_test.sample(n=min(3, len(df_test)), random_state=None)
            default_images = []
            for _, row in sampled.iterrows():
                img_path = row['image_path']
                print(img_path)
                if os.path.exists(img_path):
                    print("Found image path")
                    try:
                        img = Image.open(img_path).convert("RGB")
                        # Create a file-like name for display
                        filename = os.path.basename(img_path)
                        default_images.append((img, filename))
                    except Exception as e:
                        print(f"Could not load {img_path}: {e}")
            return default_images
    except Exception as e:
        print(f"Could not load default images: {e}")
    return []


st.set_page_config(page_title="Brain Tumor Image Segmentation", 
                   page_icon=os.path.join(os.getcwd(), 'web_based', 'page_icon.png'),
                   layout="wide")

# Initialize session state for default images
if 'default_images' not in st.session_state:
    st.session_state.default_images = None
if 'last_use_default' not in st.session_state:
    st.session_state.last_use_default = False

st.title("Brain Tumor Image Segmentation Demo")
st.markdown("### Upload up to 5 images (PNG/JPG/JPEG) (or check `Load 3 sample images` to use sample images), then click `Run inference` to get segmentation masks.")

st.markdown("Note: In web deployment, no cuda supported")
# Sidebar controls
with st.sidebar:
    st.image(os.path.join(os.getcwd(), 'web_based', 'sidebar_icon.png'), width='stretch')
    st.header("Settings")
    model_path = st.selectbox("Model", options=["UNet", "SwinHAFNet"])
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5)
    use_default = st.checkbox("Load 3 sample images", value=False)
    run_button = st.button("Run inference")

# File uploader (allow multiple)
uploaded_files = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
if uploaded_files is None:
    uploaded_files = []

# Handle default images with session state
# Load new random images only when checkbox state changes from unchecked to checked
if use_default:
    if not st.session_state.last_use_default:
        # Checkbox just got checked, load new random images
        st.session_state.default_images = load_default_images()
        st.session_state.last_use_default = True
    default_images = st.session_state.default_images or []
    if default_images and len(uploaded_files) == 0:
        st.info(f"Loaded {len(default_images)} sample images")
else:
    # Checkbox is unchecked, clear state
    st.session_state.last_use_default = False
    st.session_state.default_images = None
    default_images = []

if len(uploaded_files) > 5:
    st.warning("Please upload at most 5 images.")
    uploaded_files = uploaded_files[:5]

# Process uploaded files
images = []
image_names = []
for i, up in enumerate(uploaded_files):
    try:
        image = Image.open(io.BytesIO(up.read())).convert("RGB")
        images.append(image)
        image_names.append(up.name)
    except Exception as e:
        st.error(f"Failed to read {up.name}: {e}")

# Add default images if using defaults and no uploads
if use_default and len(images) == 0:
    for img, name in default_images:
        images.append(img)
        image_names.append(name)
# Display images
if len(images) > 0:
    cols = st.columns(min(5, len(images)))
    for i, (img, name) in enumerate(zip(images, image_names)):
        cols[i].image(img, width='stretch', caption=name)

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
    st.warning("No images uploaded. Please upload images or check 'Load 3 sample images'.")

st.markdown("---")