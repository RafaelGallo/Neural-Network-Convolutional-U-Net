import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Path to model (make sure your .h5 is in the 'models' folder)
MODEL_PATH = os.path.join("models", "unet_2_segmentacao.h5")

@st.cache_resource
def load_unet_model():
    model = load_model(MODEL_PATH)
    return model

model = load_unet_model()

st.title("Lung Lesion Segmentation (U-Net)")
st.write("Upload a CT or X-ray image (PNG/JPG). Optionally, upload the ground truth mask to compute segmentation metrics (Accuracy, Dice, IoU).")

# Upload CT/X-ray image
uploaded_file = st.file_uploader("Upload clinical image (X-ray or CT)", type=["png", "jpg", "jpeg"])

# Optional: Upload ground truth mask
uploaded_mask = st.file_uploader("Upload Ground Truth Mask (optional)", type=["png", "jpg", "jpeg"])

def is_medical_xray(img_pil):
    # Accept grayscale or almost grayscale images (basic heuristic)
    img_np = np.array(img_pil)
    if img_np.ndim == 2:  # Already grayscale
        return True
    if img_np.ndim == 3:
        # Check if all channels are very similar
        dif = np.abs(img_np[:,:,0] - img_np[:,:,1]) + np.abs(img_np[:,:,0] - img_np[:,:,2])
        mean_dif = np.mean(dif)
        if mean_dif < 5:
            return True
    return False

def dice_coef_np(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_coef_np(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

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

        # Display results
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

        # If the user uploaded a ground truth mask, calculate metrics
        if uploaded_mask is not None:
            mask_true = Image.open(uploaded_mask).convert("L").resize((256, 256))
            mask_true = np.array(mask_true)
            mask_true_bin = (mask_true > 127).astype(np.uint8)
            
            # Metrics
            accuracy_pixel = np.mean(mask_true_bin == mask_pred_bin)
            dice = dice_coef_np(mask_true_bin, mask_pred_bin)
            iou = iou_coef_np(mask_true_bin, mask_pred_bin)

            st.success(f"**Segmentation accuracy (pixel):** {accuracy_pixel:.4f}")
            st.success(f"**Dice coefficient:** {dice:.4f}")
            st.success(f"**IoU:** {iou:.4f}")

            # Display masks side by side
            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
            axs2[0].imshow(mask_true_bin, cmap='gray')
            axs2[0].set_title('Ground Truth Mask')
            axs2[0].axis('off')
            axs2[1].imshow(mask_pred_bin, cmap='gray')
            axs2[1].set_title('Predicted Mask (Binary)')
            axs2[1].axis('off')
            st.pyplot(fig2)
        else:
            st.info("Upload the ground truth mask (PNG/JPG) to see segmentation metrics like accuracy, Dice, and IoU.")
