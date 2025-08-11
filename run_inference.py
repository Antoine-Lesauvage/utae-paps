import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F

#fold = 1
with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

with open("NORM_S2_patch.json") as f:
    norm_vals = json.load(f)
 
# Convertir le dict en objet (similaire à un namespace)
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Dataset lecture S3 --------
class S3SinglePatchDataset(Dataset):
    def __init__(self, s3_prefix, tile_ids, transform=None):
        self.s3_prefix = s3_prefix.rstrip("/")
        self.tile_ids = tile_ids
        self.transform = transform
        os.environ["AWS_ACCESS_KEY_ID"] = '1CH53NXGI0M8EML05AQY'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'YCz+C1hM3kokhoJTmISBFV84VHqpgmw4dUZ6Y5pk'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiIxQ0g1M05YR0kwTThFTUwwNUFRWSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0Mzc5NDA3LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU0OTg0MzUzLCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDM3OTU1MiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDpmNTBmMDZkNC1kZWU1LWE4N2QtY2FhNC01MTg5NTkyZDVjZWQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJiODNkZWZiNC1kOWQwLTRhNWYtYjM3Ny1jMjI5NGZiYmMwZGEiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.v9xea_8hrBrC-X5PDukdL3hvn96XkN1cbmu9fHIlHd9_Hhe3OD68tjcCpfXdK22AvoKrrRCFseRjfyCwKzm--w'
        os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
        self.fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        path = f"{self.s3_prefix}/{tile_id}.npy"
        with self.fs.open(path, 'rb') as f:
            arr = np.load(f)
            # Convertir en tensor PyTorch
        tensor = torch.from_numpy(arr).float()  # (T, C, H, W)

    # Appliquer padding/cropping
        #tensor = pad_or_crop(tensor, target_h=128, target_w=128) #AL commented out
        if self.transform:
            tensor = self.transform(tensor)
        return {"tile_id": tile_id, "data": tensor}  # (T, C, H, W)

# -------- Chargement modèle --------
from src.model_utils import get_model

def load_model(ckpt_path):
    model = model_utils.get_model(config, mode="panoptic")
    model.to(device)

# Charger les poids du modèle
    #checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold}", "model.pth.tar")
    assert os.path.exists(ckpt_path), f"Fichier introuvable : {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device)
#print("📦 Clés disponibles dans le checkpoint :", checkpoint.keys())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.cuda()

# -------- Inférence --------
def run_inference():
    s3_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128/"
    tile_ids = ["0", "1"]  # exemple

    dataset = S3SinglePatchDataset(s3_prefix, tile_ids)
    dataloader = DataLoader(dataset, batch_size=3)

    # 🔁 Charger tous les modèles et normalisations
    models = []
    means = []
    stds = []
    for fold_id in range(1,6):
        ckpt_path = os.path.join("../UTAE_PAPs", f"Fold_{fold_id}", "model.pth.tar")
        model = load_model(ckpt_path)
        models.append(model)

        mean = torch.tensor(norm_vals[f"Fold_{fold_id}"]["mean"]).float().cuda()
        std = torch.tensor(norm_vals[f"Fold_{fold_id}"]["std"]).float().cuda()
        means.append(mean)
        stds.append(std)

    for batch in dataloader:
        tile_batch = batch["tile_id"]
        datas = batch["data"]  # shape: [B, T, C, H, W]
        for idx, (tile_id, data) in enumerate(zip(tile_batch, datas)):
            print(f"Traitement de la tuile {tile_id}")
            x_orig = data.unsqueeze(0).cuda()  # shape: [1, T, C, H, W]
            T, C, H, W = x_orig.shape[1:]
            batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()

            semantic_logits_accum = None

            # 🔁 Moyenne sur les 5 folds
            for model, mean, std in zip(models, means, stds):
                mean = mean.reshape(1, -1, 1, 1).to(x_orig.device)
                std = std.reshape(1, -1, 1, 1).to(x_orig.device)

                x = (x_orig - mean) / std
                print(x.shape)  # avant passage au modèle
                with torch.no_grad():
                    output = model(x, batch_positions=batch_positions)
                    logits = output["semantic"]  # shape: [1, nb_classes, H, W]
                    print(f"Fold {fold_id} → logits shape: {logits.shape}")
                    

                    if semantic_logits_accum is None:
                        semantic_logits_accum = logits
                    else:
                        print(f"Shape logits: {logits.shape}, accumulated: {semantic_logits_accum.shape if semantic_logits_accum is not None else None}")
                        #semantic_logits_accum += logits

            # 🔁 Moyenne
            
            semantic_logits_avg = semantic_logits_accum / 5
            semantic_pred = semantic_logits_avg.argmax(dim=1).cpu().numpy()  # (1, H, W)

            # Sauvegarde
            np.save(f"preds/{tile_id}_pred_avg.npy", semantic_pred)
            print(f"✅ Tuile {tile_id} traitée avec moyenne des folds")


if __name__ == "__main__":
    run_inference()