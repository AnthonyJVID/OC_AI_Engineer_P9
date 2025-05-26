# fonctions.py

#from config import DATA_DIR, RESULTS_DIR

# -------------------- FONCTIONS DE BASE DATANT DU PROJET 8 --------------------

# fonctions.py

# Importations nécessaires
import os
import tensorflow as tf
from cityscapesscripts.helpers.labels import name2label
from cityscapesscripts.preparation.json2labelImg import json2labelImg
import json
import numpy as np
import albumentations as A
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from albumentations import Compose, HorizontalFlip, Rotate, OneOf, RandomScale, Blur, GaussNoise, Resize
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, Resizing, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tqdm import tqdm
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from cityscapesscripts.helpers.labels import trainId2label
import time
import segmentation_models as sm
import pandas as pd
from pathlib import Path
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import glob
import torch
from typing import Tuple
from torchvision import transforms
import torch.nn.functional as F


# Définition des classes utiles
CLASSES_UTILES = {
    "void": 0, "flat": 1, "construction": 2, "object": 3,
    "nature": 4, "sky": 5, "human": 6, "vehicle": 7
}

# Correction du chemin pour Projet 9
root_path = Path(".")  # racine du projet 9
data_path = root_path / "data"
cityscapes_scripts_path = root_path / "notebook/cityscapesScripts/cityscapesscripts"
images_path = data_path / "leftImg8bit"
masks_path = data_path / "gtFine"

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", mode="fine", target_type="semantic", image_size=(512, 512)):
        from torchvision.datasets import Cityscapes
        from torchvision import transforms
        self.dataset = Cityscapes(root=root, split=split, mode="fine", target_type="semantic")
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size)

        # Convertir l’image en tenseur
        image = self.transforms.ToTensor()(image)

        # Convertir le masque en tableau numpy puis appliquer le remapping
        mask_np = np.array(mask).astype(np.uint8)
        mask_remap = remap_classes(mask_np)

        mask_tensor = torch.from_numpy(mask_remap).long()
        return image, mask_tensor

def remap_classes(mask: np.ndarray) -> np.ndarray:
    """
    Convertit les classes Cityscapes originales (0-33) vers les 8 catégories principales définies.
    Retourne un masque avec uniquement des valeurs de 0 à 7.
    """

    # Nettoyage des valeurs non prévues (ex: 34, 35)
    mask = np.where(mask > 33, 0, mask)  # Toute valeur > 33 est convertie en void (classe 0)

    # Définition précise du mapping basé sur les "labelIds" Cityscapes originaux
    labelIds_to_main_classes = {
        0: 0,   # unlabeled → void
        1: 0,   # ego vehicle → void
        2: 0,   # rectification border → void
        3: 0,   # out of roi → void
        4: 0,   # static → void
        5: 0,   # dynamic → void
        6: 0,   # ground → void
        7: 1,   # road → flat
        8: 1,   # sidewalk → flat
        9: 0,   # parking → void
        10: 0,  # rail track → void
        11: 2,  # building → construction
        12: 2,  # wall → construction
        13: 2,  # fence → construction
        14: 0,  # guard rail → void
        15: 0,  # bridge → void
        16: 0,  # tunnel → void
        17: 3,  # pole → object
        18: 3,  # polegroup → object
        19: 3,  # traffic light → object
        20: 3,  # traffic sign → object
        21: 4,  # vegetation → nature
        22: 4,  # terrain → nature
        23: 5,  # sky → sky
        24: 6,  # person → human
        25: 6,  # rider → human
        26: 7,  # car → vehicle
        27: 7,  # truck → vehicle
        28: 7,  # bus → vehicle
        29: 7,  # caravan → vehicle
        30: 7,  # trailer → vehicle
        31: 7,  # train → vehicle
        32: 7,  # motorcycle → vehicle
        33: 7   # bicycle → vehicle
    }

    remapped_mask = np.copy(mask)
    for original_class, new_class in labelIds_to_main_classes.items():
        remapped_mask[mask == original_class] = new_class

    return remapped_mask.astype(np.uint8)


def view_folder(dossier):
    dossier = Path(dossier)
    if not dossier.exists():
        print(f"❌ Le dossier {dossier} n'existe pas.")
        return
    for sous_dossier in dossier.iterdir():
        if sous_dossier.is_dir():
            print(f"|-- {sous_dossier.name}")
            for sous_sous_dossier in sous_dossier.iterdir():
                if sous_sous_dossier.is_dir():
                    print(f"    |-- {sous_sous_dossier.name}")

def load_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Charge et normalise une image entre 0 et 1."""
    img = load_img(path, target_size=target_size)
    return img_to_array(img).astype("float32") / 255.0

def load_mask(path: str, target_size: Tuple[int, int], mask_mode="labelIds") -> np.ndarray:
    """
    Charge, redimensionne et remappe un masque.
    Applique systématiquement le remapping vers les 8 classes principales.

    Args:
        path (str): Chemin vers le masque.
        target_size (Tuple[int, int]): Taille de sortie (hauteur, largeur).
        mask_mode (str): "labelIds" pour les masques Cityscapes originaux, "trainIds" sinon.

    Returns:
        np.ndarray: Masque avec valeurs de classe entre 0 et 7.
    """
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    mask = img_to_array(mask).astype("uint8").squeeze()

    # Toujours appliquer le remapping pour garantir 8 classes
    mask = remap_classes(mask)

    return mask

def one_hot_encode_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Encode un masque en One-Hot."""

    # Vérifier les valeurs uniques avant l'encodage
    unique_values = np.unique(mask)
    if np.any(unique_values >= num_classes):
        print(f"Attention : Certaines valeurs de masques dépassent {num_classes-1}: {unique_values}")
        mask = np.clip(mask, 0, num_classes - 1)

    return np.eye(num_classes, dtype=np.uint8)[mask]

def decode_mask(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque One-Hot en format indexé."""
    return np.argmax(mask, axis=-1)

def get_augmentations(image_size: Tuple[int, int]) -> Compose:
    """Définit les transformations Albumentations pour l'entraînement."""
    return Compose([
        HorizontalFlip(p=0.2),
        Rotate(limit=15, p=0.2),
        RandomScale(scale_limit=0.1, p=0.2),
        Resize(*image_size, interpolation=cv2.INTER_NEAREST)
    ])

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, image_size=(256, 256), batch_size=16, num_classes=8, # TEST avec 512x512, 1024x1024, 512x1024, 1024x512, 256x512 et 512x256
                shuffle=True, augmentation_ratio=1.0, use_cache=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation_ratio = augmentation_ratio
        self.use_cache = use_cache
        self.cache = {}  # Cache des masques transformés
        self.augmentation = get_augmentations(image_size)
        self.on_epoch_end()

    def __getitem__(self, index):
        start_time = time.time()
        start = index * self.batch_size
        end = start + self.batch_size
        batch_image_paths = self.image_paths[start:end]
        batch_mask_paths = self.mask_paths[start:end]

        batch_images, batch_masks = [], []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            img = load_image(img_path, self.image_size)

            if self.use_cache and mask_path in self.cache:
                mask = self.cache[mask_path]
            else:
                mask = load_mask(mask_path, self.image_size, mask_mode="trainIds")
                if self.use_cache:
                    self.cache[mask_path] = mask

            if np.random.rand() < self.augmentation_ratio:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            batch_images.append(img)
            batch_masks.append(one_hot_encode_mask(mask, self.num_classes))

        elapsed_time = time.time() - start_time
        # print(f"📊 Génération batch {index} en {elapsed_time:.2f}s")

        return np.stack(batch_images), np.stack(batch_masks)

    def __len__(self):
        """Renvoie le nombre total de batches par epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Mélange les données après chaque epoch si shuffle est activé."""
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        """Affiche correctement un lot d'images et de masques."""
        batch_images, batch_masks = self.__getitem__(0)
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(decode_mask(batch_masks[i]), cmap="inferno")
            axes[i, 1].set_title("Mask (decoded)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()


# Test du DataGenerator
if __name__ == "__main__":
    train_gen = DataGenerator(
        image_paths=train_input_img_paths,
        mask_paths=train_label_ids_img_paths,
        image_size=(256, 256),  # TEST avec 512x512
        batch_size=16,  # TEST: 8, 16 ou 32
        num_classes=8,
        shuffle=True,
        augmentation_ratio=0.5
    )

    train_gen.visualize_batch(num_images=3)

    def on_epoch_end(self) -> None:
        """Mélange les données après chaque epoch si shuffle est activé."""
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        """Affiche correctement un lot d'images et de masques."""
        batch_images, batch_masks = self.__getitem__(0)
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(decode_mask(batch_masks[i]), cmap="inferno")
            axes[i, 1].set_title("Mask (decoded)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

def iou_coef(y_true, y_pred, smooth=1e-6):
    """
    Calcule l'Intersection over Union (IoU).
    Correction : conversion explicite en float32.
    """
    y_true = tf.keras.backend.cast(y_true, "float32")
    y_pred = tf.keras.backend.cast(y_pred, "float32")
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)



def get_logger(nom_modele: str):
    """
    Crée un CSVLogger pour enregistrer les métriques d'entraînement dans un fichier horodaté.
    """
    from datetime import datetime
    from tensorflow.keras.callbacks import CSVLogger

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = RESULTS_DIR / f"{nom_modele}_{timestamp}.csv"

    return CSVLogger(log_filename, separator=",", append=False)

def charger_metriques(dossier_logs):
    """
    Charge tous les fichiers CSV de métriques présents dans un dossier.

    Args:
        dossier_logs (str): Chemin vers le dossier contenant les fichiers CSV.

    Returns:
        dict: Dictionnaire avec nom du modèle en clé et dataframe en valeur.
    """
    fichiers = glob.glob(os.path.join(dossier_logs, "*.csv"))
    resultats = {}

    for fichier in fichiers:
        # Récupère le nom complet du modèle (par exemple unet_mini, unet_vgg16)
        nom_modele = "_".join(os.path.basename(fichier).split("_")[:-2])
        df = pd.read_csv(fichier)
        resultats[nom_modele] = df

    return resultats

def tracer_metriques(resultats):
    """
    Trace les métriques des différents modèles sur des graphiques.

    Args:
        resultats (dict): Dictionnaire avec nom modèle et dataframe.
    """

    # Palette de couleurs spécifique pour chaque modèle
    couleurs = {
        "mini": "blue",
        "vgg16": "green",
        "resnet50": "red",
        "efficientnetb3": "purple"
    }

    plt.figure(figsize=(18, 18))

    # Graphique de Loss (Perte)
    plt.subplot(3, 2, 1)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        plt.plot(df["loss"], label=f"{modele} Train Loss", color=couleur, linestyle="--")
        plt.plot(df["val_loss"], label=f"{modele} Val Loss", color=couleur, linestyle="-")
    plt.title("Comparaison des Loss (Perte)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Graphique Mean IoU
    plt.subplot(3, 2, 2)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "mean_iou" in df.columns:
            plt.plot(df["mean_iou"], label=f"{modele} Train Mean IoU", color=couleur, linestyle="--")
            plt.plot(df["val_mean_iou"], label=f"{modele} Val Mean IoU", color=couleur, linestyle="-")
        elif "iou_score" in df.columns:
            plt.plot(df["iou_score"], label=f"{modele} Train IoU Score", color=couleur, linestyle="--")
            plt.plot(df["val_iou_score"], label=f"{modele} Val IoU Score", color=couleur, linestyle="-")
    plt.title("Comparaison du Mean IoU / IoU Score")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.legend()

    # Graphique Dice Coefficient
    plt.subplot(3, 2, 3)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "dice_coef" in df.columns:
            plt.plot(df["dice_coef"], label=f"{modele} Train Dice", color=couleur, linestyle="--")
            plt.plot(df["val_dice_coef"], label=f"{modele} Val Dice", color=couleur, linestyle="-")
    plt.title("Comparaison du Dice Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.grid(True)
    plt.legend()

    # Graphique Accuracy
    plt.subplot(3, 2, 4)
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "accuracy" in df.columns:
            plt.plot(df["accuracy"], label=f"{modele} Train Accuracy", color=couleur, linestyle="--")
            plt.plot(df["val_accuracy"], label=f"{modele} Val Accuracy", color=couleur, linestyle="-")
    plt.title("Comparaison de l'Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Graphique Temps d'entraînement par modèle
    plt.subplot(3, 1, 3)
    temps_entrainement = {}
    for modele, df in resultats.items():
        couleur = couleurs.get(modele, "black")
        if "temps_total_sec" in df.columns:
            temps = df["temps_total_sec"].iloc[-1] / 60  # converti en minutes
            temps_entrainement[modele] = temps
            plt.bar(modele, temps, color=couleur)
            plt.text(modele, temps, f"{temps:.2f} min", ha="center", va="bottom")

    plt.title("Comparaison du Temps total d'entraînement (en minutes)")
    plt.ylabel("Temps (minutes)")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

# -------------------- NOUVELLES FONCTIONS POUR PROJET 9 --------------------

def charger_oneformer(num_classes: int = 8):
    """
    Charge le modèle OneFormer adapté au dataset Cityscapes.
    """
    from transformers import OneFormerForSemanticSegmentation
    model = OneFormerForSemanticSegmentation.from_pretrained("nvidia/oneformer_coco_swin_large")
    model.config.num_labels = num_classes
    return model


def charger_segnext(num_classes: int = 8):
    """
    Charge le modèle SegNeXt-L (simplifié avec timm ou autre wrapper).
    """
    import timm
    model = timm.create_model("segnext_l", pretrained=True, num_classes=num_classes)
    return model

def entrainer_model_pytorch(
    model,
    train_loader,
    val_loader,
    model_name="model",
    epochs=10,
    lr=1e-4,
    num_classes=8
):
    """
    Entraîne un modèle PyTorch de segmentation avec :
    - Mixed Precision (torch.cuda.amp)
    - GradScaler pour la stabilité
    - Scheduler 'ReduceLROnPlateau'
    - Gestion de la sortie pour SegFormer (SemanticSegmenterOutput)
    ou un simple tenseur
    - Upsampling de la sortie pour correspondre au masque (H, W)
    - Calcul et log des métriques (accuracy, Dice, IoU) pour train et val
    - Mesure du temps par epoch et de la mémoire GPU peak
    - Sauvegarde CSV + .pth dans '../resultats_modeles/'
    - Génération d'un graphique PNG de l'évolution du Dice et du Mean IoU.
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_sched
    from torch.cuda.amp import autocast, GradScaler
    from transformers.modeling_outputs import SemanticSegmenterOutput
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import time
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Définition locale des métriques PyTorch (évite doublons) --------
    def compute_batch_metrics(pred_logits, target, num_classes):
        """
        Calcule accuracy, Dice et IoU moyens (macro) pour un batch.
        - pred_logits: (N, C, H, W)
        - target: (N, H, W) (valeurs entières [0..num_classes-1])
        Retourne un dict: {"accuracy": float, "dice": float, "iou": float}
        """
        # 1) Conversion argmax => (N, H, W)
        pred = torch.argmax(pred_logits, dim=1)

        # 2) Accuracy globale (tous pixels confondus)
        correct = (pred == target).sum().item()
        total = target.numel()  # N*H*W
        accuracy = correct / total

        # 3) Intersection / union par classe => Dice, IoU
        dice_list = []
        iou_list = []

        for c in range(num_classes):
            pred_c = (pred == c)
            target_c = (target == c)

            inter = (pred_c & target_c).sum().item()
            pred_area = pred_c.sum().item()
            target_area = target_c.sum().item()
            union = pred_area + target_area - inter

            # IoU
            if union == 0:
                # classe absente dans les 2 => convention IoU = 1
                iou_c = 1.0
            else:
                iou_c = inter / union

            # Dice = 2*inter / (|pred_c| + |target_c|)
            denom = pred_area + target_area
            if denom == 0:
                dice_c = 1.0
            else:
                dice_c = 2.0 * inter / denom

            dice_list.append(dice_c)
            iou_list.append(iou_c)

        mean_dice = sum(dice_list) / len(dice_list)
        mean_iou = sum(iou_list) / len(iou_list)

        return {"accuracy": accuracy, "dice": mean_dice, "iou": mean_iou}

    # -------- Setup Optim / Loss / Scheduler / GradScaler --------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()

    os.makedirs("../resultats_modeles", exist_ok=True)

    # -------- Structure du log --------
    log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_dice_coef": [],
        "train_mean_iou": [],
        "val_accuracy": [],
        "val_dice_coef": [],
        "val_mean_iou": [],
        "epoch_time_s": [],
        "peak_gpu_mem_mb": []
    }

    start_time = time.time()

    # ============================ BOUCLE D'ENTRAÎNEMENT ============================
    for epoch in range(epochs):
        # Pour mesurer le pic de mémoire GPU sur l'epoch
        torch.cuda.reset_peak_memory_stats(device=device)
        epoch_start = time.time()

        # -------- TRAIN LOOP --------
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_dice = 0.0
        running_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outdict = model(images)
                # Gérer SegFormer / DeepLab / simple Tensor
                if isinstance(outdict, SemanticSegmenterOutput):
                    logits = outdict.logits
                elif isinstance(outdict, dict):
                    logits = outdict["out"]
                else:
                    logits = outdict

                # Upsample -> (N, C, H, W) = taille de masks
                logits = F.interpolate(
                    logits,
                    size=(masks.shape[-2], masks.shape[-1]),
                    mode='bilinear',
                    align_corners=False
                )

                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Calcul des métriques sur ce batch
            metrics_batch = compute_batch_metrics(logits, masks, num_classes=num_classes)
            running_accuracy += metrics_batch["accuracy"]
            running_dice += metrics_batch["dice"]
            running_iou += metrics_batch["iou"]

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)

        # -------- VALID LOOP --------
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_running_dice = 0.0
        val_running_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Val"):
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    outdict = model(images)
                    if isinstance(outdict, SemanticSegmenterOutput):
                        logits = outdict.logits
                    elif isinstance(outdict, dict):
                        logits = outdict["out"]
                    else:
                        logits = outdict

                    logits = F.interpolate(
                        logits,
                        size=(masks.shape[-2], masks.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )

                    loss_val = criterion(logits, masks)

                val_running_loss += loss_val.item()

                metrics_batch_val = compute_batch_metrics(logits, masks, num_classes=num_classes)
                val_running_accuracy += metrics_batch_val["accuracy"]
                val_running_dice += metrics_batch_val["dice"]
                val_running_iou += metrics_batch_val["iou"]

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_accuracy = val_running_accuracy / len(val_loader)
        avg_val_dice = val_running_dice / len(val_loader)
        avg_val_iou = val_running_iou / len(val_loader)

        # -------- Scheduler : ReduceLROnPlateau --------
        scheduler.step(avg_val_loss)

        # -------- Log de fin d’epoch --------
        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(device=device)
        peak_mem_mb = peak_mem / (1024 ** 2)

        log["epoch"].append(epoch + 1)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)
        log["train_accuracy"].append(avg_train_accuracy)
        log["train_dice_coef"].append(avg_train_dice)
        log["train_mean_iou"].append(avg_train_iou)
        log["val_accuracy"].append(avg_val_accuracy)
        log["val_dice_coef"].append(avg_val_dice)
        log["val_mean_iou"].append(avg_val_iou)
        log["epoch_time_s"].append(epoch_time)
        log["peak_gpu_mem_mb"].append(peak_mem_mb)

        print(
            f"📉 Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | "
            f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
            f"Time: {epoch_time:.1f}s | GPU: {peak_mem_mb:.1f} MB"
        )

    # ============================ FIN DE L'ENTRAÎNEMENT ============================
    total_time = time.time() - start_time

    # -------- Sauvegarde du log en CSV --------
    df = pd.DataFrame(log)
    df["temps_total_sec"] = total_time
    os.makedirs("../resultats_modeles", exist_ok=True)
    csv_path = f"../resultats_modeles/{model_name}_log.csv"
    df.to_csv(csv_path, index=False)

    # -------- Sauvegarde des poids --------
    torch.save(model.state_dict(), f"../resultats_modeles/{model_name}.pth")

    # -------- Génération et sauvegarde d'un graphique (Dice/IoU) --------
    plt.figure(figsize=(12, 5))

    # Subplot 1 : Dice
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_dice_coef"], label="Train Dice", color="blue")
    plt.plot(df["epoch"], df["val_dice_coef"], label="Val Dice", color="orange")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)

    # Subplot 2 : IoU
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_mean_iou"], label="Train IoU", color="blue")
    plt.plot(df["epoch"], df["val_mean_iou"], label="Val IoU", color="orange")
    plt.title("Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    png_path = f"../resultats_modeles/{model_name}_dice_iou.png"
    plt.savefig(png_path, dpi=100)
    plt.close()

    print(f"✅ Entraînement {model_name} terminé en {total_time:.1f} secondes.")
    print(f"📁 Logs : {csv_path}")
    print(f"📁 Modèle : ../resultats_modeles/{model_name}.pth")
    print(f"📊 Graphique Dice/IoU sauvegardé : {png_path}")

def comparer_resultats(dossier='../resultats_modeles'):
    """
    Affiche les courbes d'apprentissage de chaque modèle entraîné.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    plt.figure(figsize=(10, 6))
    for file in os.listdir(dossier):
        if file.endswith("_log.csv"):
            df = pd.read_csv(os.path.join(dossier, file))
            nom = file.replace("_log.csv", "")
            plt.plot(df["epoch"], df["train_loss"], label=f"{nom} train")
            plt.plot(df["epoch"], df["val_loss"], label=f"{nom} val")
    plt.title("Courbes d'apprentissage")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------- FONCTIONS REECRITE POUR LE PROJET 9 --------------------

def charger_donnees_cityscapes(data_dir: str, batch_size: int = 16, image_size: Tuple[int, int] = (256, 256)):
    """
    Charge les données Cityscapes et retourne deux DataLoaders (train et val).
    Utilise CityscapesDataset, et applique:
    - num_workers=4
    - pin_memory=True
    pour des perfs optimales sur GPU
    """
    from torch.utils.data import DataLoader

    train_dataset = CityscapesDataset(root=data_dir, split="train", image_size=image_size)
    val_dataset = CityscapesDataset(root=data_dir, split="val", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader

import matplotlib.patches as mpatches

# Palette colorimétrique douce (8 classes utiles)
PALETTE = {
    0: (0, 0, 0),           # void → noir
    1: (50, 50, 150),       # flat → bleu foncé
    2: (102, 0, 204),       # construction → violet
    3: (255, 85, 0),        # object → orange
    4: (255, 255, 0),       # nature → jaune
    5: (0, 255, 255),       # sky → cyan
    6: (255, 0, 255),       # human → magenta
    7: (255, 255, 255),     # vehicle → blanc
}

CLASS_NAMES = {
    0: "void",
    1: "flat",
    2: "construction",
    3: "object",
    4: "nature",
    5: "sky",
    6: "human",
    7: "vehicle"
}

def decode_cityscapes_mask(mask):
    """
    Convertit un masque 2D (valeurs de 0 à 7) en image RGB pour affichage.
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        mask_rgb[mask == class_id] = color
    return mask_rgb

def afficher_image_et_masque(image_tensor, mask_tensor):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    PALETTE = [
        (0, 0, 0),          # 0 - void
        (100, 0, 200),      # 1 - flat
        (70, 70, 70),       # 2 - construction
        (250, 170, 30),     # 3 - object
        (107, 142, 35),     # 4 - nature
        (70, 130, 180),     # 5 - sky
        (220, 20, 60),      # 6 - human
        (0, 0, 142),        # 7 - vehicle
    ]
    PALETTE_NP = np.array(PALETTE) / 255.0
    cmap = ListedColormap(PALETTE_NP)

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im = plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=7)
    cbar = plt.colorbar(im, ticks=range(8))
    cbar.ax.set_yticklabels(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    cbar.set_label("Catégories", rotation=270, labelpad=15)
    plt.title("Masque (8 classes colorisées)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def charger_segformer(num_classes=8):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=8,
        ignore_mismatched_sizes=True
    )
    model.config.num_labels = num_classes
    model.config.output_hidden_states = False
    return model

def charger_deeplabv3plus(num_classes=8):
    import torchvision.models.segmentation as models
    import torch.nn as nn

    model = models.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

class MiniCityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Charger l’image et le masque
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Charger l’image
        from PIL import Image
        image = Image.open(image_path).convert("RGB").resize(self.image_size)

        # Charger le masque
        mask = Image.open(mask_path).convert("L").resize(self.image_size)

        # Convertir en tenseur PyTorch
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        image = to_tensor(image)  # shape (3, H, W)

        # Numpy + remap classes
        import numpy as np
        mask_np = np.array(mask, dtype=np.uint8)

        # Remap
        mask_np = remap_classes(mask_np)
        mask_tensor = torch.from_numpy(mask_np).long()  # shape (H, W)

        return image, mask_tensor

def show_predictions(model, dataset, num_images=3, num_classes=8):
    """
    Affiche quelques prédictions vs masques réels depuis un dataset PyTorch.
    Gère upsample, SegFormer / DeepLab / etc.
    """
    import torch
    import matplotlib.pyplot as plt
    from transformers.modeling_outputs import SemanticSegmenterOutput
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for i in range(num_images):
        # Choisir un index aléatoire
        idx = np.random.randint(0, len(dataset))
        image, mask_gt = dataset[idx]  # (3, H, W), (H, W)

        image_t = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        mask_gt_np = mask_gt.numpy()  # (H, W)

        with torch.no_grad():
            outdict = model(image_t)
            if isinstance(outdict, SemanticSegmenterOutput):
                logits = outdict.logits
            elif isinstance(outdict, dict):
                logits = outdict["out"]
            else:
                logits = outdict

            logits = F.interpolate(
                logits,
                size=mask_gt.shape,
                mode='bilinear',
                align_corners=False
            )
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # AFFICHAGES
        axes[i, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_gt_np, cmap="tab10", vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title("Masque GT")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="tab10", vmin=0, vmax=num_classes-1)
        axes[i, 2].set_title("Masque Prédit")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

def charger_maskformer(num_classes=8):
    """
    Charge un modèle MaskFormer (HuggingFace Transformers)
    pour la segmentation.
    S'appuie sur un checkpoint préentraîné sur ADE20K.
    """
    from transformers import MaskFormerForInstanceSegmentation

    # Exemple : "facebook/maskformer-swin-large-ade" (semantic sur ADE20K)
    # ou "facebook/maskformer-swin-base-coco" (panoptic/instance, COCO)
    # À adapter selon votre besoin.
    checkpoint = "facebook/maskformer-swin-large-ade"

    model = MaskFormerForInstanceSegmentation.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True  # parfois nécessaire si on change num_labels
    )

    # Ajuster le nombre de classes pour Cityscapes (8)
    model.config.num_labels = num_classes
    # Facultatif : désactiver l'output des hidden states
    model.config.output_hidden_states = False

    return model


import torch
import torch.nn.functional as F

def maskformer_aggregator(
    class_queries_logits: torch.Tensor,
    masks_queries_logits: torch.Tensor
) -> torch.Tensor:
    """
    Combine les prédictions de Mask(2)Former (class_queries_logits, masks_queries_logits)
    en un tenseur de forme (N, C, H, W) pour la segmentation sémantique.

    Hypothèses :
    - class_queries_logits: (N, Q, C)  [logits par classe pour chaque query]
    - masks_queries_logits: (N, Q, H, W) [logits masques (souvent à interpréter en sigmoid)]

    Approche naïve :
    1) On transforme class_queries_logits en probabilités par softmax sur la dimension 'classe' (C).
    2) On applique une sigmoïde sur masks_queries_logits pour obtenir p(query=1) par pixel.
    3) On effectue un produit de chacun de ces masques par la proba de sa classe,
        puis on somme sur la dimension 'Q' pour obtenir un tenseur (N, C, H, W).
    4) On laisse ce tenseur en l'état (non normalisé) pour que CrossEntropyLoss effectue
        son propre softmax. On l'appelle 'aggregated_logits'.

    Résultat :
    aggregated_logits.shape == (N, C, H, W),
    que vous pourrez envoyer dans F.cross_entropy(aggregated_logits, targets).
    """
    # 1) Softmax sur la dimension 'classe' => shape (N, Q, C)
    class_probs = F.softmax(class_queries_logits, dim=2)

    # 2) Sigmoïde sur la dimension 'pixel' => shape (N, Q, H, W)
    mask_probs = torch.sigmoid(masks_queries_logits)

    # 3) Produit puis somme : on fait un Einstein summation ou un broadcasting
    #    aggregated[b, c, h, w] = sum_q( class_probs[b,q,c] * mask_probs[b,q,h,w] )
    aggregated = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)

    # Ici, aggregated est un "score" par classe et par pixel, non normalisé.
    # CrossEntropyLoss attend un tenseur (N, C, H, W) de logits,
    # puis fait un log_softmax interne. aggregated étant positif, on peut
    # éventuellement l'écraser un peu. Mais on le laisse tel quel.
    return aggregated

def training_for_maskformer(
    model,
    train_loader,
    val_loader,
    model_name="maskformer",
    epochs=10,
    lr=1e-4,
    num_classes=8
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_sched
    from torch.cuda.amp import autocast, GradScaler
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import time
    import torch.nn.functional as F

    # On importe la fonction aggregator
    from fonctions import maskformer_aggregator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Métriques
    def compute_batch_metrics(pred_logits, target, nb_classes):
        pred = torch.argmax(pred_logits, dim=1)
        correct = (pred == target).sum().item()
        total = target.numel()
        accuracy = correct / total

        dice_list = []
        iou_list = []
        for c in range(nb_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            inter = (pred_c & target_c).sum().item()
            pred_area = pred_c.sum().item()
            target_area = target_c.sum().item()
            union = pred_area + target_area - inter

            iou_c = 1.0 if union == 0 else inter / union
            denom = pred_area + target_area
            dice_c = 1.0 if denom == 0 else (2.0 * inter / denom)

            dice_list.append(dice_c)
            iou_list.append(iou_c)

        mean_dice = sum(dice_list) / len(dice_list)
        mean_iou = sum(iou_list) / len(iou_list)
        return {"accuracy": accuracy, "dice": mean_dice, "iou": mean_iou}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()

    os.makedirs("../resultats_modeles", exist_ok=True)

    log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_dice_coef": [],
        "train_mean_iou": [],
        "val_accuracy": [],
        "val_dice_coef": [],
        "val_mean_iou": [],
        "epoch_time_s": [],
        "peak_gpu_mem_mb": []
    }

    start_time = time.time()

    for epoch in range(epochs):
        torch.cuda.reset_peak_memory_stats(device=device)
        epoch_start = time.time()

        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_dice = 0.0
        running_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                # outputs est de type MaskFormerForInstanceSegmentationOutput
                class_queries = outputs.class_queries_logits   # (N, Q, num_labels)
                masks_queries = outputs.masks_queries_logits    # (N, Q, h, w)

                # On upsample les masques pour correspondre à la taille des ground truth
                masks_queries = F.interpolate(
                    masks_queries,
                    size=(masks.shape[-2], masks.shape[-1]),
                    mode='bilinear',
                    align_corners=False
                )

                # On agrège en un tenseur (N, C, H, W)
                aggregated_logits = maskformer_aggregator(class_queries, masks_queries)

                loss = criterion(aggregated_logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Métriques
            metrics_batch = compute_batch_metrics(aggregated_logits, masks, num_classes)
            running_accuracy += metrics_batch["accuracy"]
            running_dice += metrics_batch["dice"]
            running_iou += metrics_batch["iou"]

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)

        # ---------------- VAL ----------------
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_running_dice = 0.0
        val_running_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Val"):
                images, masks = images.to(device), masks.to(device)

                with autocast():
                    outputs = model(images)
                    class_queries = outputs.class_queries_logits
                    masks_queries = outputs.masks_queries_logits

                    masks_queries = F.interpolate(
                        masks_queries,
                        size=(masks.shape[-2], masks.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    aggregated_logits = maskformer_aggregator(class_queries, masks_queries)

                    loss_val = criterion(aggregated_logits, masks)

                val_running_loss += loss_val.item()
                val_metrics = compute_batch_metrics(aggregated_logits, masks, num_classes)
                val_running_accuracy += val_metrics["accuracy"]
                val_running_dice += val_metrics["dice"]
                val_running_iou += val_metrics["iou"]

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_accuracy = val_running_accuracy / len(val_loader)
        avg_val_dice = val_running_dice / len(val_loader)
        avg_val_iou = val_running_iou / len(val_loader)

        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

        log["epoch"].append(epoch + 1)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)
        log["train_accuracy"].append(avg_train_accuracy)
        log["train_dice_coef"].append(avg_train_dice)
        log["train_mean_iou"].append(avg_train_iou)
        log["val_accuracy"].append(avg_val_accuracy)
        log["val_dice_coef"].append(avg_val_dice)
        log["val_mean_iou"].append(avg_val_iou)
        log["epoch_time_s"].append(epoch_time)
        log["peak_gpu_mem_mb"].append(peak_mem)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | "
            f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
            f"Time: {epoch_time:.1f}s | GPU: {peak_mem:.1f} MB"
        )

    total_time = time.time() - start_time
    df = pd.DataFrame(log)
    df["temps_total_sec"] = total_time
    csv_path = f"../resultats_modeles/{model_name}_log.csv"
    df.to_csv(csv_path, index=False)

    # Sauvegarde du modèle
    torch.save(model.state_dict(), f"../resultats_modeles/{model_name}.pth")

    # Génération d’un graphique Dice/IoU
    plt.figure(figsize=(12, 5))

    # Plot Dice
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_dice_coef"], label="Train Dice", color="blue")
    plt.plot(df["epoch"], df["val_dice_coef"], label="Val Dice", color="orange")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_mean_iou"], label="Train IoU", color="blue")
    plt.plot(df["epoch"], df["val_mean_iou"], label="Val IoU", color="orange")
    plt.title("Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    png_path = f"../resultats_modeles/{model_name}_dice_iou.png"
    plt.savefig(png_path, dpi=100)
    plt.close()

    print(f"✅ Entraînement {model_name} terminé en {total_time:.1f} secondes.")
    print(f"📁 Logs : {csv_path}")
    print(f"📁 Modèle : ../resultats_modeles/{model_name}.pth")
    print(f"📊 Graphique Dice/IoU sauvegardé : {png_path}")

def training_for_mask2former(
    model,
    train_loader,
    val_loader,
    model_name="mask2former",
    epochs=10,
    lr=1e-4,
    num_classes=8
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_sched
    from torch.cuda.amp import autocast, GradScaler
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import time
    import torch.nn.functional as F

    from fonctions import maskformer_aggregator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_batch_metrics(pred_logits, target, nb_classes):
        pred = torch.argmax(pred_logits, dim=1)
        correct = (pred == target).sum().item()
        total = target.numel()
        accuracy = correct / total

        dice_list = []
        iou_list = []
        for c in range(nb_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            inter = (pred_c & target_c).sum().item()
            pred_area = pred_c.sum().item()
            target_area = target_c.sum().item()
            union = pred_area + target_area - inter

            iou_c = 1.0 if union == 0 else inter / union
            denom = pred_area + target_area
            dice_c = 1.0 if denom == 0 else (2.0 * inter / denom)

            dice_list.append(dice_c)
            iou_list.append(iou_c)

        mean_dice = sum(dice_list) / len(dice_list)
        mean_iou = sum(iou_list) / len(iou_list)
        return {"accuracy": accuracy, "dice": mean_dice, "iou": mean_iou}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()

    os.makedirs("../resultats_modeles", exist_ok=True)

    log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_dice_coef": [],
        "train_mean_iou": [],
        "val_accuracy": [],
        "val_dice_coef": [],
        "val_mean_iou": [],
        "epoch_time_s": [],
        "peak_gpu_mem_mb": []
    }

    start_time = time.time()

    for epoch in range(epochs):
        torch.cuda.reset_peak_memory_stats(device=device)
        epoch_start = time.time()

        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_dice = 0.0
        running_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                # outputs est de type Mask2FormerForUniversalSegmentationOutput
                class_queries = outputs.class_queries_logits   # (N, Q, num_labels)
                masks_queries = outputs.masks_queries_logits    # (N, Q, h, w)

                masks_queries = F.interpolate(
                    masks_queries,
                    size=(masks.shape[-2], masks.shape[-1]),
                    mode='bilinear',
                    align_corners=False
                )

                aggregated_logits = maskformer_aggregator(class_queries, masks_queries)
                loss = criterion(aggregated_logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            metrics_batch = compute_batch_metrics(aggregated_logits, masks, num_classes)
            running_accuracy += metrics_batch["accuracy"]
            running_dice += metrics_batch["dice"]
            running_iou += metrics_batch["iou"]

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)

        # ---------------- VAL ----------------
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_running_dice = 0.0
        val_running_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Val"):
                images, masks = images.to(device), masks.to(device)

                with autocast():
                    outputs = model(images)
                    class_queries = outputs.class_queries_logits
                    masks_queries = outputs.masks_queries_logits

                    masks_queries = F.interpolate(
                        masks_queries,
                        size=(masks.shape[-2], masks.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    aggregated_logits = maskformer_aggregator(class_queries, masks_queries)

                    loss_val = criterion(aggregated_logits, masks)

                val_running_loss += loss_val.item()
                val_metrics = compute_batch_metrics(aggregated_logits, masks, num_classes)
                val_running_accuracy += val_metrics["accuracy"]
                val_running_dice += val_metrics["dice"]
                val_running_iou += val_metrics["iou"]

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_accuracy = val_running_accuracy / len(val_loader)
        avg_val_dice = val_running_dice / len(val_loader)
        avg_val_iou = val_running_iou / len(val_loader)

        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

        log["epoch"].append(epoch + 1)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_val_loss)
        log["train_accuracy"].append(avg_train_accuracy)
        log["train_dice_coef"].append(avg_train_dice)
        log["train_mean_iou"].append(avg_train_iou)
        log["val_accuracy"].append(avg_val_accuracy)
        log["val_dice_coef"].append(avg_val_dice)
        log["val_mean_iou"].append(avg_val_iou)
        log["epoch_time_s"].append(epoch_time)
        log["peak_gpu_mem_mb"].append(peak_mem)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | "
            f"Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f} | "
            f"Time: {epoch_time:.1f}s | GPU: {peak_mem:.1f} MB"
        )

    total_time = time.time() - start_time
    df = pd.DataFrame(log)
    df["temps_total_sec"] = total_time
    csv_path = f"../resultats_modeles/{model_name}_log.csv"
    df.to_csv(csv_path, index=False)
    torch.save(model.state_dict(), f"../resultats_modeles/{model_name}.pth")

    # Génération courbes Dice/IoU
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_dice_coef"], label="Train Dice", color="blue")
    plt.plot(df["epoch"], df["val_dice_coef"], label="Val Dice", color="orange")
    plt.title("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_mean_iou"], label="Train IoU", color="blue")
    plt.plot(df["epoch"], df["val_mean_iou"], label="Val IoU", color="orange")
    plt.title("Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    png_path = f"../resultats_modeles/{model_name}_dice_iou.png"
    plt.savefig(png_path, dpi=100)
    plt.close()

    print(f"✅ Entraînement {model_name} terminé en {total_time:.1f} secondes.")
    print(f"📁 Logs : {csv_path}")
    print(f"📁 Modèle : ../resultats_modeles/{model_name}.pth")
    print(f"📊 Graphique Dice/IoU sauvegardé : {png_path}")

def show_predictions_maskformer(
    model,
    dataset,
    num_images=3,
    num_classes=8
):
    """
    Affiche quelques prédictions vs masques réels depuis un dataset PyTorch,
    pour un modèle MaskFormer-like (avec class_queries_logits et masks_queries_logits).

    1) On récupère `class_queries_logits` et `masks_queries_logits`.
    2) On upsample le masks_queries_logits à la taille du masque target.
    3) On agrège via maskformer_aggregator pour obtenir un tenseur (N, C, H, W).
    4) On calcule un argmax (H, W) pour l'affichage.
    """

    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.cuda.amp import autocast
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # On importe la fonction aggregator déjà définie
    # (celle qui combine class_queries_logits et masks_queries_logits)
    from fonctions import maskformer_aggregator

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for i in range(num_images):
        idx = np.random.randint(0, len(dataset))
        image, mask_gt = dataset[idx]  # (3, H, W), (H, W)

        image_t = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        mask_gt_np = mask_gt.numpy()            # (H, W)

        with torch.no_grad(), autocast():
            outputs = model(image_t)
            # Récupération des logits
            class_queries = outputs.class_queries_logits  # (1, Q, num_labels)
            masks_queries = outputs.masks_queries_logits  # (1, Q, h, w)

            # Upsample le masks_queries à la taille du mask GT
            masks_queries = F.interpolate(
                masks_queries,
                size=(mask_gt_np.shape[0], mask_gt_np.shape[1]),
                mode='bilinear',
                align_corners=False
            )

            # Agrégation => (1, C, H, W)
            aggregated_logits = maskformer_aggregator(class_queries, masks_queries)
            # Argmax => (H, W)
            pred = torch.argmax(aggregated_logits, dim=1).squeeze(0).cpu().numpy()

        # AFFICHAGE
        if num_images == 1:
            # Juste 1 image => axes est un tableau 1D [3 subplots]
            ax_img, ax_gt, ax_pred = axes
        else:
            ax_img, ax_gt, ax_pred = axes[i]

        ax_img.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax_img.set_title("Image")
        ax_img.axis("off")

        ax_gt.imshow(mask_gt_np, cmap="tab10", vmin=0, vmax=num_classes-1)
        ax_gt.set_title("Masque GT")
        ax_gt.axis("off")

        ax_pred.imshow(pred, cmap="tab10", vmin=0, vmax=num_classes-1)
        ax_pred.set_title("Masque Prédit")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import os

def comparer_modeles(list_csv_files, model_names=None):
    """
    Compare plusieurs modèles sur les métriques d'entraînement (loss, dice, iou, accuracy)
    et affiche un bar chart du temps total.

    Args:
        list_csv_files (list): liste des chemins vers les fichiers CSV de logs.
        model_names (list): noms courts à afficher en légende. Doit être de même taille que list_csv_files.
                        Si None, on utilise le nom de fichier.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if model_names is None:
        model_names = [os.path.splitext(os.path.basename(csv_file))[0] for csv_file in list_csv_files]

    # On charge chaque CSV dans un DataFrame, qu'on stocke dans un dict
    model_data = {}
    for csv_file, name in zip(list_csv_files, model_names):
        df = pd.read_csv(csv_file)
        model_data[name] = df

    # Couleurs prédéfinies pour la cohérence
    color_list = ["red", "blue", "green", "purple", "orange", "black"]
    # Création de la figure : 3 lignes, 2 colonnes → 5 subplots (le dernier occupant une ligne entière)
    fig = plt.figure(figsize=(14, 14))

    # -- SUBPLOT 1 : Loss (en haut à gauche) --
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.set_title("Comparaison des Loss (Perte)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    for i, (name, df) in enumerate(model_data.items()):
        c = color_list[i % len(color_list)]
        if "train_loss" in df.columns and "val_loss" in df.columns:
            ax1.plot(df["epoch"], df["train_loss"], label=f"{name} Train Loss", color=c, linestyle="--")
            ax1.plot(df["epoch"], df["val_loss"],   label=f"{name} Val Loss",   color=c, linestyle="-")
    ax1.grid(True)
    ax1.legend()

    # -- SUBPLOT 2 : Accuracy (en haut à droite) --
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.set_title("Comparaison de l'Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    for i, (name, df) in enumerate(model_data.items()):
        c = color_list[i % len(color_list)]
        if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
            ax2.plot(df["epoch"], df["train_accuracy"], label=f"{name} Train Acc", color=c, linestyle="--")
            ax2.plot(df["epoch"], df["val_accuracy"],   label=f"{name} Val Acc",   color=c, linestyle="-")
    ax2.grid(True)
    ax2.legend()

    # -- SUBPLOT 3 : Dice (en bas à gauche) --
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.set_title("Comparaison du Dice Coefficient")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Dice Coefficient")
    for i, (name, df) in enumerate(model_data.items()):
        c = color_list[i % len(color_list)]
        if "train_dice_coef" in df.columns and "val_dice_coef" in df.columns:
            ax3.plot(df["epoch"], df["train_dice_coef"], label=f"{name} Train Dice", color=c, linestyle="--")
            ax3.plot(df["epoch"], df["val_dice_coef"],   label=f"{name} Val Dice",   color=c, linestyle="-")
    ax3.grid(True)
    ax3.legend()

    # -- SUBPLOT 4 : Mean IoU (en bas à droite) --
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.set_title("Comparaison du Mean IoU")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Mean IoU")
    for i, (name, df) in enumerate(model_data.items()):
        c = color_list[i % len(color_list)]
        if "train_mean_iou" in df.columns and "val_mean_iou" in df.columns:
            ax4.plot(df["epoch"], df["train_mean_iou"], label=f"{name} Train IoU", color=c, linestyle="--")
            ax4.plot(df["epoch"], df["val_mean_iou"],   label=f"{name} Val IoU",   color=c, linestyle="-")
    ax4.grid(True)
    ax4.legend()

    # -- SUBPLOT 5 : Temps total (bar chart) --
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax5.set_title("Comparaison du Temps total d'entraînement (en minutes)")
    training_times = []
    for i, (name, df) in enumerate(model_data.items()):
        if "temps_total_sec" in df.columns:
            total_time_sec = df["temps_total_sec"].iloc[-1]
            total_time_min = total_time_sec / 60
        else:
            total_time_min = 0
        training_times.append((name, total_time_min))

    x_labels = [t[0] for t in training_times]
    y_values = [t[1] for t in training_times]
    bars = ax5.bar(x_labels, y_values, color=color_list[:len(y_values)])
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.2f} min",
                ha='center', va='bottom')
    ax5.set_ylabel("Temps (minutes)")
    ax5.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# FONCTIONS POUR SIMULER LA PLUIE ET COMPARER LES PRÉDICTIONS
# ------------------------------------------------------------------

import albumentations as A
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Transformation globale (effet pluie)
rain_transform = A.Compose([
    A.RandomRain(
        brightness_coefficient=0.9,
        drop_length=20,
        drop_width=1,
        blur_value=3,
        rain_type='heavy'
    )
])

def apply_rain_effect(image_pil: Image.Image) -> Image.Image:
    """
    Applique l'effet de pluie à une image PIL et renvoie une nouvelle image PIL.
    """
    # Convertir en NumPy
    image_np = np.array(image_pil)

    # Appliquer la transformation Albumentations
    augmented = rain_transform(image=image_np)
    rain_np = augmented['image']

    # Reconvertir en PIL
    rain_pil = Image.fromarray(rain_np)
    return rain_pil

def predict_mask(model, image_pil, device="cpu", num_classes=8):
    """
    Utilise 'model' (PyTorch) pour prédire le masque de l'image PIL.
    Retourne un array NumPy (H,W) avec les classes prédites [0..7].
    """
    # Conversion PIL -> Tensor
    transform = transforms.ToTensor()  # [0..1], shape (3,H,W)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        # Ex.: si c’est un SegFormer, on accède à outputs.logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs["out"]
        else:
            logits = outputs

        # Upsample => taille de l'image originale
        _, _, h_img, w_img = image_tensor.shape
        logits = F.interpolate(
            logits,
            size=(h_img, w_img),
            mode='bilinear',
            align_corners=False
        )

        # argmax => (H,W)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred_mask

def compare_rain_predictions(
    baseline_model,
    new_model,
    image_path,
    device="cpu",
    size=(256,256)
):
    """
    1) Charge l'image d'origine.
    2) Redimensionne en (size), applique la pluie.
    3) Fait prédire le masque par baseline_model et new_model.
    4) Retourne un fig (matplotlib) avec 4 colonnes :
    - image originale
    - image "pluie"
    - masque baseline
    - masque new model
    """
    # 1) Charger et redimensionner l'image
    pil_image = Image.open(image_path).convert("RGB").resize(size)

    # 2) Appliquer la pluie
    rain_pil = apply_rain_effect(pil_image)

    # 3) Prédictions
    mask_old = predict_mask(baseline_model, rain_pil, device=device)
    mask_new = predict_mask(new_model, rain_pil, device=device)

    # 4) Préparer l'affichage
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    axs[0].imshow(np.array(pil_image))
    axs[0].set_title("Original")
    axs[1].imshow(np.array(rain_pil))
    axs[1].set_title("Pluie")
    axs[2].imshow(mask_old, cmap="magma", vmin=0, vmax=7)
    axs[2].set_title("Masque (baseline)")
    axs[3].imshow(mask_new, cmap="magma", vmin=0, vmax=7)
    axs[3].set_title("Masque (nouveau)")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    return fig