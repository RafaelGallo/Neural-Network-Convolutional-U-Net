import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Caminho do modelo (ajuste se necessário)
MODEL_PATH = os.path.join("models", "unet_2_segmentacao.h5")

@st.cache_resource
def load_unet_model():
    model = load_model(MODEL_PATH)
    return model

model = load_unet_model()

st.title("Segmentação de Lesão Pulmonar com U-Net")

st.markdown("""
Envie uma imagem clínica (raio-x ou tomografia) no formato PNG ou JPG.<br>
Se você enviar também a máscara real (opcional), as métricas de acurácia, Dice e IoU serão exibidas.
""", unsafe_allow_html=True)

# Upload da imagem
uploaded_file = st.file_uploader("Envie uma imagem clínica (obrigatório)", type=["png", "jpg", "jpeg"])
# Upload opcional da máscara real
uploaded_mask = st.file_uploader("Envie a máscara real (opcional)", type=["png", "jpg", "jpeg"])

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

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if not is_medical_xray(img):
        st.error("Esta imagem não parece ser um raio-x/tomografia clínica. Envie uma imagem radiográfica ou de TC em tons de cinza.")
    else:
        img = img.convert("L")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        st.image(img, caption='Imagem Original', use_column_width=True)
        img_input = np.expand_dims(img_array, axis=(0, -1))  # (1, 256, 256, 1)

        # Predição da máscara
        mask_pred = model.predict(img_input)[0, :, :, 0]
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)

        # Visualização
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_array, cmap='gray')
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(mask_pred, cmap='gray')
        axs[1].set_title('Máscara Predita (Prob)')
        axs[1].axis('off')
        axs[2].imshow(img_array, cmap='gray')
        axs[2].imshow(mask_pred_bin, cmap='jet', alpha=0.4)
        axs[2].set_title('Máscara Predita Sobreposta')
        axs[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # Se a máscara real foi enviada, calcula métricas
        if uploaded_mask is not None:
            mask_true = Image.open(uploaded_mask).convert("L").resize((256, 256))
            mask_true = np.array(mask_true)
            mask_true_bin = (mask_true > 127).astype(np.uint8)
            
            # Métricas
            accuracy_pixel = np.mean(mask_true_bin == mask_pred_bin)
            dice = dice_coef_np(mask_true_bin, mask_pred_bin)
            iou = iou_coef_np(mask_true_bin, mask_pred_bin)

            st.markdown("""
            ### Métricas de Segmentação
            **Acurácia pixel a pixel:** {:.4f}  
            **Dice coefficient:** {:.4f}  
            **IoU:** {:.4f}
            """.format(accuracy_pixel, dice, iou))

            with st.expander("O que significam essas métricas?"):
                st.markdown("""
                - **Acurácia pixel a pixel:** Proporção de pixels corretamente classificados em relação ao total.  
                - **Dice coefficient:** Métrica de similaridade entre a máscara predita e a real. Varia de 0 (sem sobreposição) a 1 (sobreposição perfeita).  
                - **IoU (Intersection over Union):** Mede a interseção dividida pela união entre as máscaras predita e real. Varia de 0 a 1, quanto maior, melhor.
                """)

            # Visualização comparativa das máscaras (real vs predita)
            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
            axs2[0].imshow(mask_true_bin, cmap='gray')
            axs2[0].set_title('Máscara Real')
            axs2[0].axis('off')
            axs2[1].imshow(mask_pred_bin, cmap='gray')
            axs2[1].set_title('Máscara Predita')
            axs2[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Envie também a máscara real para visualizar as métricas quantitativas (Acurácia, Dice, IoU).")
