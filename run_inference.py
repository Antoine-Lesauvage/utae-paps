import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F
def pad_or_crop(tensor, target_h, target_w):
    _, _, h, w = tensor.shape
    dh, dw = target_h - h, target_w - w

    # Crop if too big
    if dh < 0 or dw < 0:
        tensor = tensor[:, :, :min(h, target_h), :min(w, target_w)]

    # Pad if too small
    if dh > 0 or dw > 0:
        pad = [0, max(0, dw), 0, max(0, dh)]  # pad right, pad bottom
        tensor = F.pad(tensor, pad, mode='constant', value=0)

    return tensor

fold = 2
with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

# Convertir le dict en objet (similaire à un namespace)
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Dataset lecture S3 --------
class S3SinglePatchDataset(Dataset):
    def __init__(self, s3_prefix, tile_ids, transform=None):
        self.s3_prefix = s3_prefix.rstrip("/")
        self.tile_ids = tile_ids
        self.transform = transform
        os.environ["AWS_ACCESS_KEY_ID"] = 'YD0TAHNZGMEOI88NBN94'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'pb8cmrFPZsfanu6mUQgzEoxboE9xZT2pdN+RTVir'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJZRDBUQUhOWkdNRU9JODhOQk45NCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzUzODU5NjU2LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU0NDY1OTM1LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1Mzg2MTEzNSwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDoyMjA1ZDZhMS0xNGY5LTQ0YWYtYTc4ZS04ZTZhOWE1MTYwNjciLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJkMzgyNzQ3OS1iZmY3LTRmNDQtOTY2Ny0wOGI2OGRjZGZlNmQiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.gd1eU-53sCcXXOyp4Oa8rxNZzThzNdMgOo-gGoTbXT-uhSCS2qw-Vd3LdIW2MV8qq4CCAeyw7_P0dNPtqX7ndw'
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
    checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold + 1}", "model.pth.tar")
    assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location=device)
#print("📦 Clés disponibles dans le checkpoint :", checkpoint.keys())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.cuda()

# -------- Inférence --------
def run_inference():
    #s3_prefix = "s3://antoinelesauvage/vergers-france/patches_2018"
    s3_prefix = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS/DATA_S2/"
    tile_ids = ["S2_10000", "S2_10001"]  # exemple, à remplacer 
    # Charger les poids du modèle
    checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_3", "model.pth.tar")


    dataset = S3SinglePatchDataset(s3_prefix, tile_ids)
    dataloader = DataLoader(dataset, batch_size=1) #changer batch_se=iez si erreur

    model = load_model(checkpoint_path)

    for batch in dataloader:
        tile_batch = batch["tile_id"]
        datas = batch["data"]  # shape: [B, T, C, H, W]
        for idx, (tile_id, data) in enumerate(zip(tile_batch, datas)):
            print(f"Traitement de la tuile {tile_id}")
            x = data.unsqueeze(0).cuda()  # shape: [1, T, C, H, W]
            T, C, H, W = x.shape[1:]
            new_H = H - (H % 32)
            new_W = W - (W % 32)
            #x = F.interpolate(x.view(-1, C, H, W), size=(new_H, new_W), mode='bilinear', align_corners=False)
            #x = x.view(1, T, C, new_H, new_W)

            # ➕ Générer les positions temporelles normalisées
            batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()  # (1, T)
            with torch.no_grad():
                output = model(x, batch_positions=batch_positions) #batch_positions
                print(f"Output keys: {output.keys()}")  # Vérifier les clés des prédictions
                if "semantic" in output:
                    semantic_logits = output["semantic"]
                    semantic_pred = semantic_logits.argmax(dim=1).cpu().numpy()  # shape: (1, H, W)
                else:
                    semantic_pred = output.get("pano_semantic", None)
                    if semantic_pred is not None:
                        semantic_pred = semantic_pred.cpu().numpy()

    # Sauvegarde (localement ici, à adapter si tu veux stocker sur S3)
            
                #np.save(f"preds/{tile_id}_pred.npy", output.cpu().numpy())
                to_save = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in output.items()}
                to_save["manual_semantic_argmax"] = semantic_pred
                
                np.save(f"preds/{tile_id}_full_output_test.npy", to_save)
                print(f"✅ Tuile {tile_id} traitée ")

if __name__ == "__main__":
    run_inference()
