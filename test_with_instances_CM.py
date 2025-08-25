import os
import sys
import json
import pandas as pd
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter src/ au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src import model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate, get_ntrainparams

# === CONFIGURATION ===
fold = 0
with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CHARGER DATASET ET MODÈLE ===
print("🧠 Chargement du modèle")
model = model_utils.get_model(config, mode="panoptic")
model.to(device)

checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold + 1}", "model.pth.tar")
assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

print("📚 Chargement des données Fold", fold + 1)
dataset = PASTIS_Dataset(
    folder=config.dataset_folder,
    target="instance",
    folds=[fold+1],
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=pad_collate,
    drop_last=True
)

# 📊 Initialisation métriques et loss
criterion = PaPsLoss(
    l_center=config.l_center,
    l_size=config.l_size,
    l_shape=config.l_shape,
    l_class=config.l_class,
    beta=config.beta,
)
meter = PanopticMeter(num_classes=config.num_classes)

# 🔄 Listes pour collecter les prédictions et vraies étiquettes
all_predictions = []
all_true_labels = []

# Classes valides (excluant 0 et 19 selon votre config)
VALID_CLASSES = list(range(1, 19))  # Classes 1 à 18
print(f"📊 Classes valides pour l'évaluation: {VALID_CLASSES}")

# 🔁 Boucle d'évaluation
print("✅ Début du test")
with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
        (inputs_data, inputs_dates), targets = batch

        inputs_data = inputs_data.to(device)
        inputs_dates = inputs_dates.to(device)
        targets = targets.to(device)

        outputs = model(inputs_data, batch_positions=inputs_dates)
        loss = criterion(outputs, targets)
        meter.add(outputs, targets)
        
        # 📊 Collecte des données pour la matrice de confusion
        try:
            # Utiliser pano_semantic
            if 'pano_semantic' in outputs:
                pano_semantic_logits = outputs['pano_semantic']  # [1, 20, 128, 128]
                pred_classes = torch.argmax(pano_semantic_logits, dim=1)  # [1, 128, 128]
            else:
                print("❌ 'pano_semantic' non trouvé dans les outputs")
                continue
            
            # UTILISER LE CANAL 6 pour les vraies étiquettes (celui avec les classes 0-19)
            if len(targets.shape) == 4:  # [B, H, W, C]
                true_classes = targets[:, :, :, 6]  # [B, H, W] - Canal 6 !
            else:
                print(f"❌ Forme inattendue pour targets: {targets.shape}")
                continue
            
            if pred_classes.shape != true_classes.shape:
                print(f"❌ Formes incompatibles: pred {pred_classes.shape} vs true {true_classes.shape}")
                continue
            
            # Convertir en numpy et aplatir
            pred_flat = pred_classes.cpu().numpy().flatten().astype(int)
            true_flat = true_classes.cpu().numpy().flatten().astype(int)
            
            # Debug pour les premiers batches
            if i < 3:
                print(f"\n📊 Batch {i} - Canal 6:")
                unique_true = np.unique(true_flat)
                unique_pred = np.unique(pred_flat)
                print(f"  Classes vraies: {unique_true}")
                print(f"  Classes prédites: {unique_pred}")
                
                # Distribution des vraies classes
                for val in unique_true:
                    count = np.sum(true_flat == val)
                    percentage = count / len(true_flat) * 100
                    if percentage > 1.0:  # Seulement si > 1%
                        print(f"    Classe {val}: {count} pixels ({percentage:.1f}%)")
            
            # Filtrer les pixels valides (garder seulement les vraies étiquettes des classes 1-18)
            valid_mask = np.isin(true_flat, VALID_CLASSES)
            
            if np.sum(valid_mask) > 0:
                all_predictions.extend(pred_flat[valid_mask])
                all_true_labels.extend(true_flat[valid_mask])
                
                if i < 3:
                    print(f"  Pixels valides: {np.sum(valid_mask)} / {len(valid_mask)}")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement du batch {i}: {e}")
            continue

# Vérifier qu'on a collecté des données
if len(all_predictions) == 0:
    print("❌ Aucune donnée collectée pour la matrice de confusion")
    sys.exit(1)

print(f"\n📊 Données collectées: {len(all_predictions)} pixels")

# Statistiques finales
all_predictions = np.array(all_predictions, dtype=int)
all_true_labels = np.array(all_true_labels, dtype=int)

unique_true = np.unique(all_true_labels)
unique_pred = np.unique(all_predictions)

print(f"📊 Classes vraies dans les données: {unique_true}")
print(f"📊 Classes prédites dans les données: {unique_pred}")

# Distribution des vraies classes
print(f"\n📊 Distribution des vraies classes:")
for classe in sorted(unique_true):
    count = np.sum(all_true_labels == classe)
    percentage = count / len(all_true_labels) * 100
    class_name = ["Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
                  "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
                  "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
                  "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
                  "Sorghum", "Void"][classe] if classe < 20 else f"Class_{classe}"
    print(f"  Classe {classe} ({class_name}): {count} pixels ({percentage:.2f}%)")

# 📈 Métriques panoptiques
sq, rq, pq = meter.value(per_class=True)
class_ids = meter.class_list

all_class_names = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
    "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
    "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers",
    "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
    "Sorghum", "Void"
]

# Conversion et sauvegarde des métriques panoptiques
SQ = sq.squeeze().cpu().numpy()
RQ = rq.squeeze().cpu().numpy()
PQ = pq.squeeze().cpu().numpy()

if len(SQ.shape) == 0:
    SQ = np.array([SQ])
if len(RQ.shape) == 0:
    RQ = np.array([RQ])
elif len(RQ.shape) > 1:
    RQ = RQ.flatten()
if len(PQ.shape) == 0:
    PQ = np.array([PQ])

class_names_evaluated = [all_class_names[i] if i < len(all_class_names) else f"Class_{i}" for i in class_ids]

df = pd.DataFrame({
    "class_id": class_ids,
    "class_name": class_names_evaluated,
    "SQ": SQ,
    "RQ": RQ,
    "PQ": PQ
})

out_csv = os.path.join(config.res_dir, f"Fold_{fold+1}", "metrics_per_class.csv")
df.to_csv(out_csv, index=False)
print(f"✅ Métriques par classe enregistrées dans {out_csv}")

# 🔥 CRÉATION DE LA MATRICE DE CONFUSION
print("\n📊 Création de la matrice de confusion...")

# Classes pour la matrice de confusion (inclure toutes les classes présentes)
all_present_classes = sorted(list(set(unique_true) | set(unique_pred)))
print(f"📊 Classes pour la matrice de confusion: {all_present_classes}")

# Calculer la matrice de confusion
cm = confusion_matrix(all_true_labels, all_predictions, labels=all_present_classes)

# Noms des classes
display_names = [all_class_names[i] if i < len(all_class_names) else f"Class_{i}" for i in all_present_classes]

# DataFrame de la matrice de confusion
cm_df = pd.DataFrame(cm, 
                     index=[f"True_{name}" for name in display_names], 
                     columns=[f"Pred_{name}" for name in display_names])

# Sauvegarder
cm_csv_path = os.path.join(config.res_dir, f"Fold_{fold+1}", "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print(f"✅ Matrice de confusion sauvegardée dans {cm_csv_path}")

# Visualisation
plt.figure(figsize=(max(12, len(all_present_classes)), max(10, len(all_present_classes))))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[name[:12] + "..." if len(name) > 12 else name for name in display_names],
            yticklabels=[name[:12] + "..." if len(name) > 12 else name for name in display_names],
            annot_kws={'size': 9})
plt.title(f'Matrice de Confusion - Fold {fold+1}\n(Classes valides: 1-18)', fontsize=14)
plt.xlabel('Prédictions', fontsize=12)
plt.ylabel('Vraies étiquettes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

cm_fig_path = os.path.join(config.res_dir, f"Fold_{fold+1}", "confusion_matrix.png")
plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Matrice de confusion visualisée et sauvegardée dans {cm_fig_path}")

# Rapport de classification
from sklearn.metrics import classification_report

report = classification_report(all_true_labels, all_predictions, 
                              labels=all_present_classes,
                              target_names=display_names,
                              output_dict=True,
                              zero_division=0)

report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(config.res_dir, f"Fold_{fold+1}", "classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"✅ Rapport de classification sauvegardé dans {report_csv_path}")

print("🎉 Analyse terminée !")