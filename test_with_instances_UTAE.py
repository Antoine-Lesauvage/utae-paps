# test_with_instances.py
import os
import sys
import json
import pandas as pd
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ajouter src/ au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src import model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate, get_ntrainparams

# === CONFIGURATION ===
fold = 2
with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

# Convertir le dict en objet (similaire à un namespace)
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CHARGER DATASET ET MODÈLE ===

print("🧠 Chargement du modèle")
model = model_utils.get_model(config, mode="panoptic")
model.to(device)

# Charger les poids du modèle
checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold + 1}", "model.pth.tar")
assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"

checkpoint = torch.load(checkpoint_path, map_location=device)
#print("📦 Clés disponibles dans le checkpoint :", checkpoint.keys())
model.load_state_dict(checkpoint["state_dict"])
model.eval()


# 📂 Chargement du dataset
print("📚 Chargement des données Fold", fold + 1)
dataset = PASTIS_Dataset(
    folder=config.dataset_folder,
    target="instance",
    folds=[fold+1],
)
dataloader = DataLoader(
    dataset,
    batch_size=2,
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

# 🔁 Boucle d'évaluation
print("✅ Début du test")
with torch.no_grad():
    for batch in tqdm(dataloader):
        (inputs_data, inputs_dates), targets = batch


# On passe les deux (data, dates) sur GPU si besoin
        inputs_data = inputs_data.to(device)
        inputs_dates = inputs_dates.to(device)
        targets = targets.to(device)

# Regroupe dans un dict comme attendu par le modèle
        outputs = model(inputs_data, batch_positions=inputs_dates)
        loss = criterion(outputs, targets)

        meter.add(outputs, targets)
        sq, rq, pq = meter.value(per_class=True)
        print("Scores par classe :")
        print("SQ :", sq)
        print("RQ :", rq)
        print("PQ :", pq)

# 📈 Affichage des résultats
sq, rq, pq = meter.value(per_class=True)
class_ids = meter.class_list  # liste des IDs de classes évaluées


# Si tu as les noms de classes, remplace-les ici
class_names = [
"Meadow",
"Soft winter wheat",
"Corn",
"Winter barley",
"Winter rapeseed",
"Spring barley",
"Sunflower",
"Grapevine",
"Beet",
 "Winter triticale",
 "Winter durum wheat",
 "Fruits,  vegetables, flowers",
 "Potatoes",
 "Leguminous fodder",
 "Soybeans",
 "Orchard",
 "Mixed cereal",
 "Sorghum"]



# Si ce n’est pas déjà le cas
SQ = sq.squeeze().cpu().numpy()
RQ = rq.squeeze().cpu().numpy()
PQ = pq.squeeze().cpu().numpy()


df = pd.DataFrame({
    "class_id": class_ids,
    "class_name": class_names,
    "SQ": SQ,
    "RQ": RQ,
    "PQ": PQ
})


# --- Enregistrement CSV ---
out_csv = os.path.join(config.res_dir, f"Fold_{fold+1}", "metrics_per_class.csv")
df.to_csv(out_csv, index=False)
print(f"✅ Métriques par classe enregistrées dans {out_csv}")