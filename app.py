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

# Language selector
LANG = st.sidebar.selectbox("Language / Idioma", ["Português", "English"])

# Texts
texts = {
    "title": {
        "Português": "Segmentação de Lesão Pulmonar com U-Net",
        "English": "Lung Lesion Segmentation with U-Net"
    },
    "intro": {
        "Português": "Envie uma imagem clínica (raio-x ou tomografia) no formato PNG ou JPG. Se você enviar também a máscara real (opcional), as métricas de acurácia, Dice e IoU serão exibidas.",
        "English": "Upload a clinical image (X-ray or CT) in PNG or JPG format. If you also provide the ground truth mask (optional), accuracy, Dice and IoU metrics will be shown."
    },
    "upload_image": {
        "Português": "Envie uma imagem clínica (obrigatório)",
        "English": "Upload a clinical image (required)"
    },
    "upload_mask": {
        "Português": "Envie a máscara real (opcional)",
        "English": "Upload ground truth mask (optional)"
    },
    "not_xray": {
        "Português": "Esta imagem não parece ser um raio-x/tomografia clínica. Envie uma imagem radiográfica ou de TC em tons de cinza.",
        "English": "This image does not seem to be a clinical X-ray or CT. Please upload a grayscale radiograph or CT image."
    },
    "original_image": {
        "Português": "Imagem Original",
        "English": "Original Image"
    },
    "mask_pred_prob": {
        "Português": "Máscara Predita (Probabilidade)",
        "English": "Predicted Mask (Probability)"
    },
    "mask_pred_overlay": {
        "Português": "Máscara Predita Sobreposta",
        "English": "Predicted Mask Overlay"
    },
    "metrics_title": {
        "Português": "Métricas de Segmentação",
        "English": "Segmentation Metrics"
    },
    "accuracy": {
        "Português": "Acurácia pixel a pixel",
        "English": "Pixel-wise Accuracy"
    },
    "dice": {
        "Português": "Dice coefficient",
        "English": "Dice coefficient"
    },
    "iou": {
        "Português": "IoU",
        "English": "IoU"
    },
    "metrics_desc": {
        "Português": """
- **Acurácia pixel a pixel:** Proporção de pixels corretamente classificados em relação ao total.
- **Dice coefficient:** Métrica de similaridade entre a máscara predita e a real. Varia de 0 (sem sobreposição) a 1 (sobreposição perfeita).
- **IoU (Intersection over Union):** Mede a interseção dividida pela união entre as máscaras predita e real. Varia de 0 a 1, quanto maior, melhor.
""",
        "English": """
- **Pixel-wise Accuracy:** Proportion of correctly classified pixels over the total.
- **Dice coefficient:** Similarity metric between predicted and ground truth masks. Ranges from 0 (no overlap) to 1 (perfect overlap).
- **IoU (Intersection over Union):** Measures intersection over union between predicted and ground truth masks. Ranges from 0 to 1, the higher the better.
"""
    },
    "mask_true": {
        "Português": "Máscara Real",
        "English": "Ground Truth Mask"
    },
    "mask_pred_bin": {
        "Português": "Máscara Predita",
        "English": "Predicted Mask"
    },
    "metrics_info": {
        "Português": "Envie também a máscara real para visualizar as métricas quantitativas (Acurácia, Dice, IoU).",
        "English": "Upload the ground truth mask to see the quantitative metrics (Accuracy, Dice, IoU)."
    }
}

# Utility functions
def is_medical_xray(img_pil):
    img_np = np.array(img_pil)
    if img_np.ndim == 2:
        return True
    if img_np.ndim == 3:
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

# Layout
st.title(texts["title"][LANG])
st.markdown(texts["intro"][LANG])

uploaded_file = st.file_uploader(texts["upload_image"][LANG], type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader(texts["upload_mask"][LANG], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if not is_medical_xray(img):
        st.error(texts["not_xray"][LANG])
    else:
        img = img.convert("L")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        st.image(img, caption=texts["original_image"][LANG], use_column_width=True)
        img_input = np.expand_dims(img_array, axis=(0, -1))  # (1, 256, 256, 1)

        mask_pred = model.predict(img_input)[0, :, :, 0]
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)

        # Visualização
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_array, cmap='gray')
        axs[0].set_title(texts["original_image"][LANG])
        axs[0].axis('off')
        axs[1].imshow(mask_pred, cmap='gray')
        axs[1].set_title(texts["mask_pred_prob"][LANG])
        axs[1].axis('off')
        axs[2].imshow(img_array, cmap='gray')
        axs[2].imshow(mask_pred_bin, cmap='jet', alpha=0.4)
        axs[2].set_title(texts["mask_pred_overlay"][LANG])
        axs[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # Métricas se tiver máscara real
        if uploaded_mask is not None:
            mask_true = Image.open(uploaded_mask).convert("L").resize((256, 256))
            mask_true = np.array(mask_true)
            mask_true_bin = (mask_true > 127).astype(np.uint8)

            accuracy_pixel = np.mean(mask_true_bin == mask_pred_bin)
            dice = dice_coef_np(mask_true_bin, mask_pred_bin)
            iou = iou_coef_np(mask_true_bin, mask_pred_bin)

            st.markdown(f"""
### {texts["metrics_title"][LANG]}
**{texts["accuracy"][LANG]}:** {accuracy_pixel:.4f}  
**{texts["dice"][LANG]}:** {dice:.4f}  
**{texts["iou"][LANG]}:** {iou:.4f}
""")
            with st.expander("ℹ️ " + ("Descrição das métricas" if LANG == "Português" else "Metrics description")):
                st.markdown(texts["metrics_desc"][LANG])

            # Máscaras comparativas
            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
            axs2[0].imshow(mask_true_bin, cmap='gray')
            axs2[0].set_title(texts["mask_true"][LANG])
            axs2[0].axis('off')
            axs2[1].imshow(mask_pred_bin, cmap='gray')
            axs2[1].set_title(texts["mask_pred_bin"][LANG])
            axs2[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info(texts["metrics_info"][LANG])
