import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained U-Net model
@st.cache_resource
def load_unet_model():
    model = load_model(r'models\unet_2_segmentacao.h5')
    return model

model = load_unet_model()

st.title("Lung Lesion Segmentation (U-Net)")
st.write("Upload a CT or X-ray image (PNG/JPG) and see the predicted segmentation mask.")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image, resize to 256x256, grayscale, normalize
    img = Image.open(uploaded_file).convert("L")
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
