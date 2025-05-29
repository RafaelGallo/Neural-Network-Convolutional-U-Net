import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Path to model
MODEL_PATH = os.path.join("models", "unet_2_segmentacao.h5")

@st.cache_resource
def load_unet_model():
    model = load_model(MODEL_PATH)
    return model

model = load_unet_model()

st.title("Lung Lesion Segmentation (U-Net)")
st.write("Upload a CT or X-ray image (PNG/JPG) and see the predicted segmentation mask.")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

def is_medical_xray(img_pil):
    # Accept only grayscale or almost grayscale images
    img_np = np.array(img_pil)
    if img_np.ndim == 2:  # Already grayscale
        return True
    if img_np.ndim == 3:
        # Check if all channels are very similar (low color variance)
        dif = np.abs(img_np[:,:,0] - img_np[:,:,1]) + np.abs(img_np[:,:,0] - img_np[:,:,2])
        mean_dif = np.mean(dif)
        if mean_dif < 5:  # threshold may be tuned
            return True
    return False

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if not is_medical_xray(img):
        st.error("This does not appear to be a clinical X-ray/CT image. Please upload a grayscale radiography or CT scan.")
    else:
        img = img.convert("L")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0

        st.image(img, caption='Original Image', use_column_width=True)

        img_input = np.expand_dims(img_array, axis=(0, -1))  # (1, 256, 256, 1)

        # Prediction
        mask_pred = model.predict(img_input)[0, :, :, 0]
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)

        # Show results
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_array, cmap='gray')
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(mask_pred, cmap='gray')
        axs[1].set_title('Predicted Mask (Prob)')
        axs[1].axis('off')
        axs[2].imshow(img_array, cmap='gray')
        axs[2].imshow(mask_pred_bin, cmap='jet', alpha=0.4)
        axs[2].set_title('Predicted Mask Overlay')
        axs[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
