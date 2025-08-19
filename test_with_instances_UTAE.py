# test_with_instances.py
import os
import sys
import json
import pandas as pd
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ajouter src/ au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src import model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate, get_ntrainparams

# === CONFIGURATION ==
fold = 0
with open("/home/onyxia/work/UATE_zenodo/conf.json") as f:
    config_dict = json.load(f)

# Convertir le dict en objet (similaire à un namespace)
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CHARGER DATASET ET MODÈLE ===

print("🧠 Chargement du modèle")
model = model_utils.get_model(config, mode="semantic")
model.to(device)

# Charger les poids du modèle
checkpoint_path = os.path.join("/home/onyxia/work/UATE_zenodo", f"Fold_{fold + 1}", "model.pth.tar")
assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 📂 Chargement du dataset
print("📚 Chargement des données Fold", fold + 1)
dataset = PASTIS_Dataset(
    folder=config.dataset_folder,
    target="semantic",
    folds=[fold+1],
)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=pad_collate,
    drop_last=True
)

# === ÉVALUATION ===
num_classes = config.num_classes
intersection = torch.zeros(num_classes, dtype=torch.float64)
union = torch.zeros(num_classes, dtype=torch.float64)
true_pos = torch.zeros(num_classes, dtype=torch.float64)
false_pos = torch.zeros(num_classes, dtype=torch.float64)
false_neg = torch.zeros(num_classes, dtype=torch.float64)

# 📊 Initialisation pour la matrice de confusion
all_preds = []
all_targets = []

print("✅ Début du test")
total_samples = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        (inputs_data, inputs_dates), targets = batch
        inputs_data = inputs_data.to(device)
        inputs_dates = inputs_dates.to(device)
        targets = targets.to(device)

        # 🔍 Debug pour le premier batch
        if batch_idx == 0:
            print(f"🔍 Dimensions input: {inputs_data.shape}")
            print(f"🔍 Dimensions targets: {targets.shape}")
            print(f"🔍 Type targets: {targets.dtype}")
            print(f"🔍 Range targets: min={targets.min()}, max={targets.max()}")
            print(f"🔍 Unique values targets: {torch.unique(targets)}")

        outputs = model(inputs_data, batch_positions=inputs_dates)
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]
        
        # ✅ Traitement adaptatif des targets selon leur format
        if len(targets.shape) == 4:  # [B, H, W, C] - format one-hot
            print("📌 Format détecté: One-hot encoding")
            targets_cls = targets.argmax(dim=-1)  # [B, H, W]
        elif len(targets.shape) == 3 and targets.dtype == torch.long:  # [B, H, W] - indices de classe
            print("📌 Format détecté: Indices de classe")
            targets_cls = targets
        elif len(targets.shape) == 3 and targets.dtype == torch.float:  # [B, H, W] - float à convertir
            print("📌 Format détecté: Float à convertir en indices")
            targets_cls = targets.long()
        else:
            print(f"⚠️  Format non reconnu: shape={targets.shape}, dtype={targets.dtype}")
            # Tentative de conversion par défaut
            if targets.dim() > 3:
                targets_cls = targets.argmax(dim=-1)
            else:
                targets_cls = targets.long()

        total_samples += targets_cls.numel()
        
        # 📊 Collecter les prédictions et cibles pour la matrice de confusion
        # Aplatir les tenseurs et les convertir en numpy
        preds_flat = preds.cpu().numpy().flatten()
        targets_flat = targets_cls.cpu().numpy().flatten()
        
        all_preds.extend(preds_flat)
        all_targets.extend(targets_flat)
        
        # 🔍 Debug pour vérifier la cohérence
        if batch_idx == 0:
            print(f"🔍 Après traitement - targets_cls shape: {targets_cls.shape}")
            print(f"🔍 Après traitement - targets_cls dtype: {targets_cls.dtype}")
            print(f"🔍 Après traitement - targets_cls range: min={targets_cls.min()}, max={targets_cls.max()}")
            print(f"🔍 Preds shape: {preds.shape}")
            print(f"🔍 Preds range: min={preds.min()}, max={preds.max()}")

        # Calcul des métriques par classe
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets_cls == cls)

            # IoU
            inter = (pred_mask & target_mask).sum().item()
            un = (pred_mask | target_mask).sum().item()
            intersection[cls] += inter
            union[cls] += un

            # TP, FP, FN pour Precision/Recall/F1
            tp = inter
            fp = (pred_mask & ~target_mask).sum().item()
            fn = (~pred_mask & target_mask).sum().item()
            true_pos[cls] += tp
            false_pos[cls] += fp
            false_neg[cls] += fn

# 📊 Calcul de la matrice de confusion
print("📊 Calcul de la matrice de confusion...")
conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

# 📊 Calcul des métriques
IoU = intersection / (union + 1e-6)
precision = true_pos / (true_pos + false_pos + 1e-6)
recall = true_pos / (true_pos + false_neg + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

# Filtrer les classes qui n'apparaissent jamais
valid_classes = union > 0
print(f"📊 Classes avec des échantillons: {valid_classes.sum()}/{num_classes}")
print(f"📊 Total échantillons traités: {total_samples}")

# Calcul des moyennes uniquement sur les classes valides
mIoU = IoU[valid_classes].mean().item() if valid_classes.sum() > 0 else 0.0
mPrecision = precision[valid_classes].mean().item() if valid_classes.sum() > 0 else 0.0
mRecall = recall[valid_classes].mean().item() if valid_classes.sum() > 0 else 0.0
mF1 = f1[valid_classes].mean().item() if valid_classes.sum() > 0 else 0.0

# Noms de classes
class_names = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
    "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
    "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
    "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
    "Sorghum", "Void"
]

# Noms courts pour l'affichage de la matrice de confusion
class_names_short = [
    "BG", "Meadow", "W.Wheat", "Corn", "W.Barley",
    "W.Rapeseed", "S.Barley", "Sunflower", "Grapevine", "Beet",
    "W.Triticale", "W.D.Wheat", "Fruits/Veg", "Potatoes", "Fodder",
    "Soybeans", "Orchard", "M.Cereal", "Sorghum", "Void"
]

# 📄 Sauvegarde CSV avec information sur les classes valides
df = pd.DataFrame({
    "class_id": range(num_classes),
    "class_name": class_names[:num_classes],
    "IoU": IoU.cpu().numpy(),
    "Precision": precision.cpu().numpy(),
    "Recall": recall.cpu().numpy(),
    "F1-score": f1.cpu().numpy(),
    "Valid": valid_classes.cpu().numpy(),
    "Samples": union.cpu().numpy()
})

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(os.path.join(config.res_dir, f"Fold_{fold+1}"), exist_ok=True)
out_csv = os.path.join(config.res_dir, f"Fold_{fold+1}", "semantic_metrics_per_class.csv")
df.to_csv(out_csv, index=False)

# 📊 Sauvegarde de la matrice de confusion
conf_matrix_df = pd.DataFrame(conf_matrix, 
                             index=class_names[:num_classes], 
                             columns=class_names[:num_classes])
conf_matrix_csv = os.path.join(config.res_dir, f"Fold_{fold+1}", "confusion_matrix.csv")
conf_matrix_df.to_csv(conf_matrix_csv)

# 📊 Visualisation de la matrice de confusion
def plot_confusion_matrix(conf_matrix, class_names_short, fold, save_path):
    """
    Créer et sauvegarder la visualisation de la matrice de confusion
    """
    # Normaliser la matrice de confusion (en pourcentage par ligne)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Remplacer NaN par 0
    
    # Créer deux subplots : matrice brute et normalisée
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Matrice brute
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_short, yticklabels=class_names_short,
                ax=ax1, cbar_kws={'label': 'Nombre de pixels'})
    ax1.set_title(f'Matrice de Confusion - Fold {fold+1}\n(Valeurs absolues)', fontsize=14)
    ax1.set_xlabel('Prédictions', fontsize=12)
    ax1.set_ylabel('Vérité terrain', fontsize=12)
    
    # Matrice normalisée
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names_short, yticklabels=class_names_short,
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title(f'Matrice de Confusion Normalisée - Fold {fold+1}\n(Proportions par ligne)', fontsize=14)
    ax2.set_xlabel('Prédictions', fontsize=12)
    ax2.set_ylabel('Vérité terrain', fontsize=12)
    
    # Rotation des labels pour une meilleure lisibilité
    for ax in [ax1, ax2]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Créer aussi une version simplifiée avec seulement les classes valides
    valid_indices = np.where(valid_classes.cpu().numpy())[0]
    if len(valid_indices) > 0:
        conf_matrix_valid = conf_matrix[np.ix_(valid_indices, valid_indices)]
        valid_class_names = [class_names_short[i] for i in valid_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Matrice brute (classes valides)
        sns.heatmap(conf_matrix_valid, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=valid_class_names, yticklabels=valid_class_names,
                    ax=ax1, cbar_kws={'label': 'Nombre de pixels'})
        ax1.set_title(f'Matrice de Confusion - Fold {fold+1}\n(Classes avec échantillons - Valeurs absolues)', fontsize=14)
        ax1.set_xlabel('Prédictions', fontsize=12)
        ax1.set_ylabel('Vérité terrain', fontsize=12)
        
        # Matrice normalisée (classes valides)
        conf_matrix_valid_norm = conf_matrix_valid.astype('float') / conf_matrix_valid.sum(axis=1)[:, np.newaxis]
        conf_matrix_valid_norm = np.nan_to_num(conf_matrix_valid_norm)
        
        sns.heatmap(conf_matrix_valid_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=valid_class_names, yticklabels=valid_class_names,
                    ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title(f'Matrice de Confusion Normalisée - Fold {fold+1}\n(Classes avec échantillons - Proportions)', fontsize=14)
        ax2.set_xlabel('Prédictions', fontsize=12)
        ax2.set_ylabel('Vérité terrain', fontsize=12)
        
        # Rotation des labels
        for ax in [ax1, ax2]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        valid_save_path = save_path.replace('.png', '_valid_classes.png')
        plt.savefig(valid_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return valid_save_path
    
    return None

# Générer les visualisations
conf_matrix_plot = os.path.join(config.res_dir, f"Fold_{fold+1}", "confusion_matrix.png")
valid_plot_path = plot_confusion_matrix(conf_matrix, class_names_short, fold, conf_matrix_plot)

# 📊 Affichage des résultats
print("\n" + "=" * 80)
print("📊 RÉSULTATS PAR CLASSE (Classes avec échantillons uniquement)")
print("=" * 80)
valid_df = df[df['Valid'] == True]
if len(valid_df) > 0:
    print(valid_df.to_string(index=False))
else:
    print("⚠️  Aucune classe valide trouvée!")

print("\n" + "=" * 80)
print("📊 MÉTRIQUES GLOBALES")
print("=" * 80)
print(f"📊 mIoU global (classes valides): {mIoU:.4f}")
print(f"📊 mPrecision (classes valides): {mPrecision:.4f}")
print(f"📊 mRecall (classes valides): {mRecall:.4f}")
print(f"📊 mF1-score (classes valides): {mF1:.4f}")

# 📊 Statistiques de la matrice de confusion
print("\n" + "=" * 80)
print("📊 STATISTIQUES MATRICE DE CONFUSION")
print("=" * 80)
total_pixels = conf_matrix.sum()
diagonal_sum = np.trace(conf_matrix)
overall_accuracy = diagonal_sum / total_pixels if total_pixels > 0 else 0

print(f"📊 Accuracy globale: {overall_accuracy:.4f}")
print(f"📊 Total pixels analysés: {total_pixels:,}")
print(f"📊 Pixels correctement classifiés: {diagonal_sum:,}")

# Affichage des classes les plus confondues
print("\n📊 Top 5 des confusions les plus fréquentes (hors diagonal):")
conf_matrix_no_diag = conf_matrix.copy()
np.fill_diagonal(conf_matrix_no_diag, 0)
flat_indices = np.argsort(conf_matrix_no_diag.flatten())[::-1][:5]
for i, flat_idx in enumerate(flat_indices):
    row, col = np.unravel_index(flat_idx, conf_matrix_no_diag.shape)
    if conf_matrix_no_diag[row, col] > 0:
        print(f"  {i+1}. {class_names[row]} → {class_names[col]}: {conf_matrix_no_diag[row, col]:,} pixels")

# 📊 Affichage des classes sans échantillons
invalid_df = df[df['Valid'] == False]
if len(invalid_df) > 0:
    print(f"\n⚠️  Classes sans échantillons ({len(invalid_df)}):")
    print(invalid_df[['class_id', 'class_name']].to_string(index=False))

print(f"\n✅ Métriques complètes enregistrées dans {out_csv}")
print(f"✅ Matrice de confusion enregistrée dans {conf_matrix_csv}")
print(f"✅ Visualisation matrice de confusion: {conf_matrix_plot}")
if valid_plot_path:
    print(f"✅ Visualisation classes valides: {valid_plot_path}")

# 🔍 Diagnostic supplémentaire
print("\n" + "=" * 80)
print("🔧 DIAGNOSTIC")
print("=" * 80)
print(f"🔍 Distribution des échantillons par classe:")
for idx, (class_name, samples) in enumerate(zip(class_names[:num_classes], union.cpu().numpy())):
    if samples > 0:
        print(f"  Classe {idx:2d} ({class_name[:30]:30s}): {samples:8.0f} pixels")