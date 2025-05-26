import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import cv2
from fonctions import rgb_to_class, dice_score, iou_score

# Palette de couleurs pour les 8 classes (Cityscapes remappées)
CITYSCAPES_COLORS = {
    0: (0, 0, 0),         # void (noir)
    1: (75, 0, 130),      # flat (violet foncé) ✅ validation WCAG
    2: (124, 124, 124),   # construction (gris) ✅ validation WCAG
    3: (255, 102, 0),     # object (orange) ✅ validation WCAG
    4: (34, 139, 34),     # nature (vert forêt) ✅ validation WCAG plus visible que (0, 255, 0)
    5: (0, 139, 139),     # sky (cyan foncé) ✅ validation WCAG plus lisible que (0, 255, 255)
    6: (178, 34, 34),     # human (rouge foncé) ✅ validation WCAG
    7: (0, 0, 255),       # vehicle (bleu) ✅ validation WCAG
}

# Fonction pour coloriser les masques
def colorize_mask(mask):
    """
    Colorise un masque de segmentation (format 2D, valeurs 0-7).
    Si le masque est déjà RGB (3D avec 3 canaux), il est retourné tel quel.
    """
    # Déjà colorisé : on le retourne tel quel
    if mask.ndim == 3 and mask.shape[2] == 3:
        return mask

    # Masque en niveaux de gris (2D)
    if mask.ndim == 2:
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in CITYSCAPES_COLORS.items():
            color_mask[mask == cls_id] = color
        return color_mask

    # Format invalide
    raise ValueError("Le masque doit être une image 2D (non colorisée) ou déjà RGB (3 canaux).")

# Fonction pour le prétraitement des images (réduites à 256x256)
def preprocess_image(img_pil):
    img = img_pil.resize((256, 256))
    img_arr = np.array(img) / 255.0
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    return img_arr.astype(np.float32)

import requests
from io import BytesIO

def call_efficientnet_api(image_pil):
    url = "http://15.236.146.43:8000/predict/" # API chez AWS (EC2)
    img_bytes = BytesIO()
    image_pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    response = requests.post(url, files={"image": ("image.png", img_bytes, "image/png")})
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise RuntimeError(f"EfficientNet API Error: {response.status_code} - {response.text}")

def call_segformer_api(image_pil):
    url = "https://jimsmith007-p9-api-segformer.hf.space/predict" # API chez Hugging Face (free CPU)
    img_bytes = BytesIO()
    image_pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    response = requests.post(url, files={"image": ("image.png", img_bytes, "image/png")})
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise RuntimeError(f"SegFormer API Error: {response.status_code} - {response.text}")

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Dashboard EDA Cityscapes",
    layout="wide"
)

# Fonction principale
def main():
    ##############################################################################
    # 1) EXPLORATION DES DONNÉES (EDA) - CITYSCAPES
    ##############################################################################

    # Titre EDA centré
    st.markdown(
        "<h2 style='text-align: center;'>Exploration du Dataset Cityscapes (EDA)</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Dans cette section, nous présentons quelques exemples d’images du dataset Cityscapes, leurs masques de segmentation, et un aperçu des 8 classes.</p>",
        unsafe_allow_html=True
    )

    # Texte explicatif sur Cityscapes
    st.markdown("""**Pourquoi Cityscapes ?**
    Cityscapes est un jeu de données de référence dans le domaine de la **vision par ordinateur**,
    en particulier pour des tâches de **segmentation sémantique** en milieu urbain. Il contient des
    images haute résolution prises depuis des caméras embarquées, couvrant des scènes variées
    (routes, bâtiments, véhicules, piétons, etc.).
    """)

    # Sous-titre d’images
    st.markdown(
        "<h3 style='text-align: center;'>Exemples d’images et masques</h3>",
        unsafe_allow_html=True
    )

    # Affichage de deux paires image/masque
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/exemple1.png", caption="Exemple 1 : Image d’une scène urbaine réelle")
    with col2:
        st.image("images/mask1.png", caption="Exemple 1 : Masque annoté correspondant (classes Cityscapes)")

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/exemple2.png", caption="Exemple 2 : Image d’une scène urbaine réelle")
    with col4:
        st.image("images/mask2.png", caption="Exemple 2 : Masque annoté correspondant (classes Cityscapes)")

    ##############################################################################
    # 2) LECTURE DU CSV CONTENANT image_path, mask_path, split, px_class0..7
    ##############################################################################
    try:
        df_all = pd.read_csv("cityscapes_subset.csv")
    except FileNotFoundError:
        st.error("Fichier cityscapes_subset.csv introuvable. Veuillez l'ajouter.")
        return

    st.markdown("## Aperçu du CSV")
    st.dataframe(df_all.head(10))

    ##############################################################################
    # 3) RÉPARTITION TRAIN / VAL
    ##############################################################################
    df_count = df_all.groupby("split")["image_path"].count().reset_index()
    df_count.columns = ["split", "count"]

    fig_split = px.bar(
        df_count,
        x="split",
        y="count",
        title="Répartition du nombre d'images (train / val)",
        labels={"split": "Split", "count": "Nombre d'images"}
    )
    st.plotly_chart(fig_split, use_container_width=True)

    ##############################################################################
    # 4) RÉPARTITION DES CLASSES (SOMME DES px_class0..7)
    ##############################################################################
    class_cols = [f"px_class{i}" for i in range(8)]
    sum_vals = df_all[class_cols].sum()  # Series index=px_class0..px_class7

    df_dist = pd.DataFrame({
        "Classe": range(8),
        "Pixels": sum_vals.values
    })

    # Conversion pour mapping discret
    df_dist["Classe_str"] = df_dist["Classe"].astype(str)

    # Définir le mapping exact (en hexadécimal)
    color_map = {
        "0": "#000000",
        "1": "#4B0082",
        "2": "#7C7C7C",
        "3": "#FF6600",
        "4": "#228B22",  # vert forêt corrigé WCAG
        "5": "#008B8B",  # cyan foncé corrigé WCAG
        "6": "#B22222",  # rouge foncé corrigé WCAG
        "7": "#0000FF"
    }


    fig_classes = px.bar(
        df_dist,
        x="Classe_str",
        y="Pixels",
        title="Répartition globale des classes par nombre de pixels (train+val)",
        labels={"Classe_str": "Classe (0..7)", "Pixels": "Nombre total de pixels"},
        color="Classe_str",
        color_discrete_map=color_map,
        category_orders={"Classe_str": [str(i) for i in range(8)]}
    )
    st.plotly_chart(fig_classes, use_container_width=True)

    ##############################################################################
    # 5) LÉGENDE DES CLASSES
    ##############################################################################
    legend_html = """
    <div style="margin-top: 15px; color: white; font-size: 20px; display: flex; justify-content: center;">
    <div style="display: flex; flex-direction: column; gap: 10px; align-items: center;">

        <!-- Ligne 1 -->
        <ul style="display: flex; list-style: none; padding: 0; gap: 30px; margin: 0; justify-content: center;">
        <li style="display: flex; align-items: center;">
            <span style="background-color: #000000; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            void (0)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #4B0082; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            flat (1)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #7C7C7C; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            construction (2)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #FF6600; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            object (3)
        </li>
        </ul>

        <!-- Ligne 2 -->
        <ul style="display: flex; list-style: none; padding: 0; gap: 30px; margin: 0; justify-content: center;">
        <li style="display: flex; align-items: center;">
            <span style="background-color: #228B22; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            nature (4)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #00FFFF; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            sky (5)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #B22222; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            human (6)
        </li>
        <li style="display: flex; align-items: center;">
            <span style="background-color: #0000FF; display:inline-block; width:20px; height:20px; margin-right: 5px;"></span>
            vehicle (7)
        </li>
        </ul>
    </div>
    </div>
    """
    components.html(legend_html, height=160)

if __name__ == "__main__":
    main()

############################################################################################################
# 6) INTERACTIVITÉ : Upload image + masque → Prédictions EfficientNet + SegFormer
############################################################################################################
st.markdown("<h1 style='text-align: center;'>Comparaison des Prédictions</h1>", unsafe_allow_html=True)

st.markdown("📁 Téléversez une image au format PNG", help="Ce champ permet d'importer une image à segmenter.")
uploaded_img = st.file_uploader("Image d'entrée", type=["png"], key="img", label_visibility="collapsed")

st.markdown("📁 Ajoutez le masque réel correspondant (même résolution)", help="Utilisé pour la comparaison avec les prédictions.")
uploaded_mask = st.file_uploader("Masque réel", type=["png"], key="mask", label_visibility="collapsed")

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB").resize((256, 256))
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image d’entrée (redimensionnée)", use_container_width=True)

    if uploaded_mask:
        mask_image = Image.open(uploaded_mask).resize((256, 256))
        with col2:
            st.image(mask_image, caption="Masque réel (déjà colorisé)", use_container_width=True)

    # Appel aux APIs externes
    pred_eff_colored = None
    seg_colored = None

    try:
        pred_eff_colored = call_efficientnet_api(image)
        st.success("✅ Prédiction EfficientNetB3 récupérée via API")
    except Exception as e:
        st.error(f"❌ Erreur EfficientNet API : {e}")

    try:
        seg_colored = call_segformer_api(image)
        st.success("✅ Prédiction SegFormer B5 récupérée via API")
    except Exception as e:
        st.error(f"❌ Erreur SegFormer API : {e}")

    # Affichage côte à côte des prédictions
    if pred_eff_colored is not None and seg_colored is not None:
        # Vérifier format sortie EfficientNet
        def grayscale_to_classes(mask_np):
            if mask_np.max() > 0:
                return np.round(mask_np / (mask_np.max() / 7.0)).astype(np.uint8)
            else:
                return mask_np

        eff_np = np.array(pred_eff_colored.convert("L"))  # Force niveaux de gris
        y_pred_eff = grayscale_to_classes(eff_np)
        pred_eff_colored_visu = colorize_mask(y_pred_eff)

        # Conversion du masque SegFormer (brut 0-7 en niveaux de gris)
        seg_np = np.array(seg_colored)
        y_pred_seg = seg_np.astype(np.uint8)
        seg_colored_visu = colorize_mask(y_pred_seg)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pred_eff_colored_visu, caption="Masque prédictif généré par EfficientNetB3 sur l’image d’entrée (classes Cityscapes)", use_container_width=True)
        with col2:
            st.image(seg_colored_visu, caption="Masque prédictif généré par SegFormer B5 sur l’image d’entrée (classes Cityscapes)", use_container_width=True)

        # Comparaison métriques
        if uploaded_mask:
            st.markdown("<h2> 🎯 Évaluation manuelle (masques réels vs prédictions)</h2>", unsafe_allow_html=True)
            y_true = rgb_to_class(np.array(mask_image.convert("RGB")))

            dice_eff = dice_score(y_true, y_pred_eff)
            iou_eff = iou_score(y_true, y_pred_eff)
            dice_seg = dice_score(y_true, y_pred_seg)
            iou_seg = iou_score(y_true, y_pred_seg)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎲 Dice EfficientNetB3", f"{dice_eff:.4f}")
                st.metric("📊 IoU EfficientNetB3", f"{iou_eff:.4f}")
            with col2:
                st.metric("🎲 Dice SegFormer B5", f"{dice_seg:.4f}")
                st.metric("📊 IoU SegFormer B5", f"{iou_seg:.4f}")

            # Analyse comparative automatique
            st.markdown("<h3> 📌 Analyse comparative automatique</h3>", unsafe_allow_html=True)
            st.caption("Un Dice plus proche de 1 indique une bonne correspondance avec le masque réel.")

            if dice_eff > dice_seg and iou_eff > iou_seg:
                st.success("✅ **EfficientNetB3** fournit de meilleures prédictions que SegFormer B5 sur cette image.")
                st.markdown(f"""
                - Dice coefficient plus élevé : **{dice_eff:.4f}** vs {dice_seg:.4f}
                - IoU plus élevé : **{iou_eff:.4f}** vs {iou_seg:.4f}
                Cela suggère que, pour cette image, **EfficientNet** parvient à mieux capturer les formes et les contours des objets.
                """)
            elif dice_seg > dice_eff and iou_seg > iou_eff:
                st.success("✅ **SegFormer B5** surpasse EfficientNetB3 sur cette image.")
                st.markdown(f"""
                - Dice coefficient plus élevé : **{dice_seg:.4f}** vs {dice_eff:.4f}
                - IoU plus élevé : **{iou_seg:.4f}** vs {iou_eff:.4f}
                Cela montre que **SegFormer** est plus performant ici, probablement grâce à son architecture basée sur les transformers.
                """)
            else:
                st.info("ℹ️ Les résultats sont mitigés : aucun modèle ne surpasse clairement l'autre.")
                st.markdown(f"""
                - Dice EfficientNet : {dice_eff:.4f} | Dice SegFormer : {dice_seg:.4f}
                - IoU EfficientNet : {iou_eff:.4f} | IoU SegFormer : {iou_seg:.4f}
                Il peut être pertinent d'analyser plus d'images pour établir une tendance fiable.
                """)

st.markdown(
    "<h2 p style='text-align:center;'>📘 Besoin d’aide sur les classes ou les masques ? <a href='https://www.cityscapes-dataset.com/dataset-overview/' target='_blank'>Voir la documentation officielle Cityscapes</a></p></h2>",
    unsafe_allow_html=True
)
