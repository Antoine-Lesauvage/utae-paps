import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F

fold = 2
with open("/home/onyxia/work/UATE_zenodo/conf.json") as f:
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
        os.environ["AWS_ACCESS_KEY_ID"] = 'VWDWXJFA2NU5TE5U03ER'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'fmTH5ymD+gcjkc+6qgrXNq+OUNTTkxj3UsyMsnNf'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJWV0RXWEpGQTJOVTVURTVVMDNFUiIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU1MDY4NDc2LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1Njc0MDk0LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NTA2OTI5NCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDo0ZWI5NWJkOC0wN2YzLTVkZTYtNmRhMy00NTc2Njc3MGQyMzgiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI4MTJjNzE3OC03YmQ5LTRjYzQtOTUxNy0yNzAyNDdhODhkODgiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.C1F-8ZPGP2jHSWgHFiAYnuhAtqPty9-2hBKyopHcnfAKDWbVg818VtWKPyRvSsGyJC8kOZ9gnsaYWybHtgUDsg'
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
    model = model_utils.get_model(config, mode="semantic")
    model.to(device)

# Charger les poids du modèle
    #checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold}", "model.pth.tar")
    assert os.path.exists(ckpt_path), f"Fichier introuvable : {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device)
#print("📦 Clés disponibles dans le checkpoint :", checkpoint.keys())
    model.load_state_dict(checkpoint["state_dict"])
    state_dict = checkpoint["state_dict"]
    # Filtrer les clés liées au panoptique si elles existent
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Exclure les couches spécifiques au panoptique
        if not any(exclude in key for exclude in ['paps', 'panoptic', 'instance']):
            filtered_state_dict[key] = value
    model.eval()
    return model.cuda()

# -------- Inférence --------
def run_inference():
    #s3_prefix = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS/DATA_S2/"
    s3_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128/"
    tile_ids = ["S2_10000","S2_30000"]  # exemple, à remplacer 
    # Charger les poids du modèle
    checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold}", "model.pth.tar")


    dataset = S3SinglePatchDataset(s3_prefix, tile_ids)
    dataloader = DataLoader(dataset, batch_size=1) #changer batch_se=iez si erreur

    model = load_model(checkpoint_path)
    mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
    std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda() 

    for batch in dataloader:
        tile_batch = batch["tile_id"]
        datas = batch["data"]  # shape: [B, T, C, H, W]
        for idx, (tile_id, data) in enumerate(zip(tile_batch, datas)):
            print(f"Traitement de la tuile {tile_id}")
            x = data.unsqueeze(0).cuda()  # shape: [1, T, C, H, W]
            if not torch.is_tensor(mean):
                mean = torch.tensor(mean, dtype=torch.float32)
            mean = mean.detach().clone().reshape(1, -1, 1, 1).to(x.device)
            if not torch.is_tensor(std):
                std = torch.tensor(std, dtype=torch.float32)
            std = std.detach().clone().reshape(1, -1, 1, 1).to(x.device)
            x = (x - mean) / std
            T, C, H, W = x.shape[1:]
            #new_H = H - (H % 32)
            #new_W = W - (W % 32)
            #x = F.interpolate(x.view(-1, C, H, W), size=(new_H, new_W), mode='bilinear', align_corners=False)
            #x = x.view(1, T, C, new_H, new_W)

            # ➕ Générer les positions temporelles normalisées
            batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()  # (1, T)
            with torch.no_grad():
                output = model(x, batch_positions=batch_positions) #batch_positions
                print(f"Output keys: {output.keys()}")  # Vérifier les clés des prédictions
                if "semantic" in output:
                    semantic_logits = output["pano_semantic"]
                    print(f"Fold {fold} → logits shape: {semantic_logits.shape}")
                    semantic_pred = semantic_logits.argmax(dim=1).cpu().numpy()  # shape: (1, H, W)
                else:
                    semantic_pred = output.get("pano_semantic", None)
                    if semantic_pred is not None:
                        semantic_pred = semantic_pred.cpu().numpy()

    # Sauvegarde (localement ici, à adapter si tu veux stocker sur S3)
            
                #np.save(f"preds/{tile_id}_pred.npy", output.cpu().numpy())
                to_save = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in output.items()}
                
                np.save(f"preds/{tile_id}_full_output_test_fold1.npy", to_save)
                print(f"✅ Tuile {tile_id} traitée ")




def load_model_fold(fold_num):
    """Charge un modèle pour un fold spécifique"""
    model = model_utils.get_model(config, mode="semantic")
    model.to(device)
    
    checkpoint_path = os.path.join("/home/onyxia/work/UATE_zenodo/", f"Fold_{fold_num}", "model.pth.tar")
    assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    state_dict = checkpoint["state_dict"]
    # Filtrer les clés liées au panoptique si elles existent
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Exclure les couches spécifiques au panoptique
        if not any(exclude in key for exclude in ['paps', 'panoptic', 'instance']):
            filtered_state_dict[key] = value
    model.eval()
    return model.cuda()

def calculate_confidence(output):
    """Calcule un score de confiance pour les prédictions"""
    if "semantic" in output:
        semantic_logits = output["semantic"]
        # Calculer l'entropie (faible entropie = haute confiance)
        probs = F.softmax(semantic_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        # Convertir en score de confiance (1 - entropie normalisée)
        max_entropy = torch.log(torch.tensor(float(probs.shape[1])))
        confidence = 1 - (entropy / max_entropy)
        return confidence.mean().item()
    return 0.0

import torch.nn.functional as F
from scipy.ndimage import zoom

def resize_prediction_to_common_size(pred, target_shape):
    """Redimensionne une prédiction à une taille cible"""
    if pred.shape == target_shape:
        return pred
    
    # Utiliser zoom de scipy pour les prédictions discrètes (classes)
    zoom_factors = [target_shape[i] / pred.shape[i] for i in range(len(target_shape))]
    resized = zoom(pred, zoom_factors, order=0)  # order=0 pour nearest neighbor
    return resized.astype(pred.dtype)

def run_inference_confidence_weighted_voting():
    """Vote pondéré par confiance avec gestion des tailles différentes"""
    #s3_prefix = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS/DATA_S2/"
    s3_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128/"
    #tile_ids = ["S2_10000","S2_30000"]  # exemple, à remplacer 
    tile_ids = ["0", "1"]
    
    dataset = S3SinglePatchDataset(s3_prefix, tile_ids)
    dataloader = DataLoader(dataset, batch_size=1)
    
    models = {}
    for fold in range(1, 6):
        models[fold] = load_model_fold(fold)
    
    for batch in dataloader:
        tile_batch = batch["tile_id"]
        datas = batch["data"]
        
        for idx, (tile_id, data) in enumerate(zip(tile_batch, datas)):
            print(f"🎯 Vote pondéré par confiance pour {tile_id}")
            x = data.unsqueeze(0).cuda()
            
            semantic_predictions = []
            confidence_maps = []
            
            for fold in range(1, 6):
                print(f"  - Fold {fold}")
                
                # Normalisation et inférence
                mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
                std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda()
                mean = mean.detach().clone().reshape(1, -1, 1, 1).to(x.device)
                std = std.detach().clone().reshape(1, -1, 1, 1).to(x.device)
                x_norm = (x - mean) / std
                
                T, C, H, W = x_norm.shape[1:]
                batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()
                
                with torch.no_grad():
                    output = models[fold](x_norm, batch_positions=batch_positions)
                    # ✅ Utiliser la fonction de traitement corrigée
                    semantic_pred, max_probs = process_model_output(output, fold)
                    if semantic_pred is not None:
                        semantic_predictions.append(semantic_pred)
                        confidence_maps.append(max_probs)
                        print(f"    ✅ Prédiction ajoutée pour fold {fold}")
                    else:
                        print(f"    ❌ Échec du traitement pour fold {fold}")
                    print(f"    ✅ Shape finale de la prédiction: {semantic_pred.shape}")
                    print(f"    Classes uniques: {np.unique(semantic_pred)[:10]}...")  # Afficher quelques classes
                    
                    # Vérifier que la prédiction a bien 2 dimensions
                    if len(semantic_pred.shape) != 2:
                        print(f"    ❌ Prédiction n'est pas 2D: {semantic_pred.shape}")
                        continue
                    
                    semantic_predictions.append(semantic_pred)
                    confidence_maps.append(max_probs)
                    print(f"    ✅ Prédiction ajoutée pour fold {fold}")
            
            print(f"📊 Nombre total de prédictions valides: {len(semantic_predictions)}")
            
            if not semantic_predictions:
                print("❌ Aucune prédiction 2D valide récupérée !")
                # Debug: essayer de sauvegarder un exemple pour analyse
                if len(tile_batch) > 0:
                    debug_output = {}
                    model_debug = models[1]
                    mean = torch.tensor(norm_vals["Fold_1"]["mean"]).float().cuda()
                    std = torch.tensor(norm_vals["Fold_1"]["std"]).float().cuda()
                    mean = mean.detach().clone().reshape(1, -1, 1, 1).to(x.device)
                    std = std.detach().clone().reshape(1, -1, 1, 1).to(x.device)
                    x_norm = (x - mean) / std
                    T, C, H, W = x_norm.shape[1:]
                    batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()
                    
                    with torch.no_grad():
                        debug_output = model_debug(x_norm, batch_positions=batch_positions)
                        debug_dict = {}
                        for key, value in debug_output.items():
                            if torch.is_tensor(value):
                                debug_dict[key] = {
                                    'shape': list(value.shape),
                                    'dtype': str(value.dtype),
                                    'min': float(value.min()),
                                    'max': float(value.max()),
                                    'sample': value.flatten()[:10].cpu().numpy().tolist()
                                }
                            else:
                                debug_dict[key] = str(value)
                        
                        np.save(f"debug_{tile_id}_output_analysis.npy", debug_dict)
                        print(f"💾 Debug sauvegardé: debug_{tile_id}_output_analysis.npy")
                continue
            
            # Suite du code (redimensionnement et vote)...
            all_shapes = [pred.shape for pred in semantic_predictions]
            print(f"📏 Toutes les shapes: {all_shapes}")
            
            max_h = max(shape[0] for shape in all_shapes)
            max_w = max(shape[1] for shape in all_shapes)
            target_shape = (max_h, max_w)
            print(f"🎯 Taille cible: {target_shape}")
            
            # Redimensionnement et vote (code existant)
            resized_predictions = []
            resized_confidences = []
            
            for i, (pred, conf) in enumerate(zip(semantic_predictions, confidence_maps)):
                print(f"  Redimensionnement Fold {i+1}: {pred.shape} -> {target_shape}")
                resized_pred = resize_prediction_to_common_size(pred, target_shape)
                resized_conf = resize_prediction_to_common_size(conf, target_shape)
                resized_predictions.append(resized_pred)
                resized_confidences.append(resized_conf)
            
            # Vote pondéré
            final_semantic = np.zeros(target_shape, dtype=semantic_predictions[0].dtype)
            
            print("🗳️ Début du vote pondéré...")
            for i in range(target_shape[0]):
                for j in range(target_shape[1]):
                    votes = [pred[i, j] for pred in resized_predictions]
                    confidences = [conf[i, j] for conf in resized_confidences]
                    
                    vote_dict = {}
                    for vote, conf in zip(votes, confidences):
                        if vote in vote_dict:
                            vote_dict[vote] += conf
                        else:
                            vote_dict[vote] = conf
                    
                    final_semantic[i, j] = max(vote_dict, key=vote_dict.get)
            
            # Sauvegarder
            result = {
                'semantic_weighted_voting': final_semantic,
                'confidence_maps': resized_confidences,
                'individual_predictions': resized_predictions,
                'original_shapes': all_shapes
            }
            np.save(f"preds/{tile_id}_weighted_voting_UTAE.npy", result)
            print(f"✅ Vote pondéré terminé pour {tile_id}")


def calculate_confidence(output):
    """Calcule un score de confiance pour les prédictions UTAE sémantique"""
    
    # ✅ Cas 1: output est directement un tensor
    if torch.is_tensor(output):
        semantic_logits = output
        if len(semantic_logits.shape) >= 3:
            # Calculer l'entropie (faible entropie = haute confiance)
            dim = 1 if len(semantic_logits.shape) == 4 else 0
            probs = F.softmax(semantic_logits, dim=dim)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
            # Convertir en score de confiance
            max_entropy = torch.log(torch.tensor(float(probs.shape[dim])))
            confidence = 1 - (entropy / max_entropy)
            return confidence.mean().item()
    
    # ✅ Cas 2: output est un dictionnaire
    elif isinstance(output, dict):
        possible_keys = ["semantic", "out", "logits", "pred", "output"]
        
        for key in possible_keys:
            if key in output:
                semantic_logits = output[key]
                if torch.is_tensor(semantic_logits) and len(semantic_logits.shape) >= 3:
                    dim = 1 if len(semantic_logits.shape) == 4 else 0
                    probs = F.softmax(semantic_logits, dim=dim)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
                    max_entropy = torch.log(torch.tensor(float(probs.shape[dim])))
                    confidence = 1 - (entropy / max_entropy)
                    return confidence.mean().item()
    
    return 0.0
def process_model_output(output, fold_num):
    """Traite les sorties du modèle UTAE sémantique"""
    
    semantic_pred = None
    max_probs = None
    
    # ✅ Cas 1: output est directement un tensor (cas courant pour UTAE sémantique)
    if torch.is_tensor(output):
        print(f"    Output est un tensor direct - Shape: {output.shape}")
        semantic_logits = output
        
        if len(semantic_logits.shape) == 4:  # (B, C, H, W)
            semantic_pred = semantic_logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)
            probs = F.softmax(semantic_logits, dim=1)[0].cpu().numpy()  # (C, H, W)
            max_probs = np.max(probs, axis=0)  # (H, W)
        elif len(semantic_logits.shape) == 3:  # (C, H, W)
            semantic_pred = semantic_logits.argmax(dim=0).cpu().numpy()  # (H, W)
            probs = F.softmax(semantic_logits, dim=0).cpu().numpy()  # (C, H, W)
            max_probs = np.max(probs, axis=0)  # (H, W)
        elif len(semantic_logits.shape) == 2:  # (H, W) - déjà des prédictions
            semantic_pred = semantic_logits.cpu().numpy()
            max_probs = np.ones_like(semantic_pred, dtype=np.float32)
        else:
            print(f"    ❌ Shape de tensor non gérée: {semantic_logits.shape}")
            return None, None
    
    # ✅ Cas 2: output est un dictionnaire
    elif isinstance(output, dict):
        print(f"    Output est un dictionnaire avec clés: {list(output.keys())}")
        possible_keys = ["semantic", "out", "logits", "pred", "output"]
        
        for key in possible_keys:
            if key in output:
                semantic_logits = output[key]
                print(f"    Utilisation de la clé '{key}' - Shape: {semantic_logits.shape}")
                
                if torch.is_tensor(semantic_logits):
                    if len(semantic_logits.shape) == 4:  # (B, C, H, W)
                        semantic_pred = semantic_logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)
                        probs = F.softmax(semantic_logits, dim=1)[0].cpu().numpy()  # (C, H, W)
                        max_probs = np.max(probs, axis=0)  # (H, W)
                        break
                    elif len(semantic_logits.shape) == 3:  # (C, H, W)
                        semantic_pred = semantic_logits.argmax(dim=0).cpu().numpy()  # (H, W)
                        probs = F.softmax(semantic_logits, dim=0).cpu().numpy()  # (C, H, W)
                        max_probs = np.max(probs, axis=0)  # (H, W)
                        break
                    elif len(semantic_logits.shape) == 2:  # (H, W) - déjà des prédictions
                        semantic_pred = semantic_logits.cpu().numpy()
                        max_probs = np.ones_like(semantic_pred, dtype=np.float32)
                        break
        
        if semantic_pred is None:
            print(f"    ❌ Aucune sortie sémantique trouvée dans: {list(output.keys())}")
    
    # ✅ Cas 3: autres types
    else:
        print(f"    ❌ Type d'output non géré: {type(output)}")
        return None, None
    
    if semantic_pred is None:
        print(f"    ❌ Impossible de traiter l'output")
        return None, None
    
    print(f"    ✅ Prédiction traitée - Shape: {semantic_pred.shape}, Classes: {len(np.unique(semantic_pred))}")
    return semantic_pred, max_probs
def calculate_confidence_utae(semantic_logits):
    """Calcule un score de confiance pour les prédictions UTAE sémantique"""
    if not torch.is_tensor(semantic_logits):
        return 0.0
    
    # Gérer différentes shapes possibles
    if len(semantic_logits.shape) == 4:  # (B, C, H, W)
        dim = 1
    elif len(semantic_logits.shape) == 3:  # (C, H, W)
        dim = 0
    else:
        return 0.0
    
    # Calculer l'entropie (faible entropie = haute confiance)
    probs = F.softmax(semantic_logits, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
    
    # Convertir en score de confiance (1 - entropie normalisée)
    max_entropy = torch.log(torch.tensor(float(probs.shape[dim]), device=probs.device))
    confidence = 1 - (entropy / max_entropy)
    return confidence.mean().item()

def process_utae_output(output):
    """Traite les sorties du modèle UTAE sémantique"""
    semantic_pred = None
    max_probs = None
    
    # UTAE sémantique retourne directement les logits de segmentation
    if torch.is_tensor(output):
        semantic_logits = output
    else:
        # Si c'est un dictionnaire, chercher la clé appropriée
        possible_keys = ["semantic", "out", "logits", "pred"]
        semantic_logits = None
        for key in possible_keys:
            if key in output:
                semantic_logits = output[key]
                break
        
        if semantic_logits is None:
            print(f"❌ Aucune sortie sémantique trouvée dans: {list(output.keys()) if isinstance(output, dict) else type(output)}")
            return None, None
    
    # Traiter les logits
    if len(semantic_logits.shape) == 4:  # (B, C, H, W)
        semantic_pred = semantic_logits.argmax(dim=1)[0].cpu().numpy()  # (H, W)
        probs = F.softmax(semantic_logits, dim=1)[0].cpu().numpy()  # (C, H, W)
        max_probs = np.max(probs, axis=0)  # (H, W)
    elif len(semantic_logits.shape) == 3:  # (C, H, W)
        semantic_pred = semantic_logits.argmax(dim=0).cpu().numpy()  # (H, W)
        probs = F.softmax(semantic_logits, dim=0).cpu().numpy()  # (C, H, W)
        max_probs = np.max(probs, axis=0)  # (H, W)
    else:
        print(f"❌ Shape non gérée: {semantic_logits.shape}")
        return None, None
    
    return semantic_pred, max_probs

def run_inference_all_folds():
    """Inférence UTAE sémantique avec tous les folds"""
    s3_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128/"
    tile_ids = ["0", "1"]  # vos tile_ids
    
    dataset = S3SinglePatchDataset(s3_prefix, tile_ids)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Charger tous les modèles UTAE
    models = {}
    for fold in range(1, 6):  # Folds 1 à 5
        print(f"📦 Chargement du modèle UTAE Fold_{fold}")
        models[fold] = load_model_fold(fold)
    
    for batch in dataloader:
        tile_batch = batch["tile_id"]
        datas = batch["data"]
        
        for idx, (tile_id, data) in enumerate(zip(tile_batch, datas)):
            print(f"🔍 Traitement UTAE de la tuile {tile_id}")
            x = data.unsqueeze(0).cuda()  # (1, T, C, H, W)
            
            # Stocker les résultats de tous les folds
            fold_results = {}
            fold_confidences = {}
            
            for fold in range(1, 6):
                print(f"  - UTAE Fold {fold}")
                
                # Normalisation spécifique au fold
                mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
                std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda()
                mean = mean.detach().clone().reshape(1, 1, -1, 1, 1).to(x.device)  # (1, 1, C, 1, 1)
                std = std.detach().clone().reshape(1, 1, -1, 1, 1).to(x.device)    # (1, 1, C, 1, 1)
                x_norm = (x - mean) / std
                
                T, C, H, W = x_norm.shape[1:]
                
                # Générer les positions temporelles pour UTAE
                batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()  # (1, T)
                
                with torch.no_grad():
                    # Appel du modèle UTAE
                    output = models[fold](x_norm, batch_positions=batch_positions)
                    
                    # Calculer la confiance
                    if torch.is_tensor(output):
                        confidence = calculate_confidence_utae(output)
                    else:
                        # Si output est un dict, prendre la première clé tensor
                        first_tensor = None
                        for v in output.values():
                            if torch.is_tensor(v):
                                first_tensor = v
                                break
                        confidence = calculate_confidence_utae(first_tensor) if first_tensor is not None else 0.0
                    
                    # Traiter et stocker les résultats
                    semantic_pred, max_probs = process_utae_output(output)
                    
                    if semantic_pred is not None:
                        fold_results[fold] = {
                            'semantic_pred': semantic_pred,
                            'confidence_map': max_probs,
                            'raw_output': output.cpu().numpy() if torch.is_tensor(output) else 
                                         {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in output.items()}
                        }
                        fold_confidences[fold] = confidence
                        print(f"    ✅ Classes prédites: {len(np.unique(semantic_pred))}, Confiance: {confidence:.4f}")
                    else:
                        print(f"    ❌ Échec du traitement pour fold {fold}")
                        fold_confidences[fold] = 0.0
            
            # Sélectionner le meilleur fold
            if fold_confidences:
                best_fold = max(fold_confidences, key=fold_confidences.get)
                print(f"✅ Meilleur fold UTAE pour {tile_id}: Fold_{best_fold} (confiance: {fold_confidences[best_fold]:.4f})")
                
                # Sauvegarder le meilleur résultat
                if best_fold in fold_results:
                    best_result = fold_results[best_fold]
                    np.save(f"preds/{tile_id}_best_fold_{best_fold}_UTAE_semantic.npy", best_result)
                
                # Sauvegarder tous les résultats pour analyse
                all_results = {
                    'fold_results': fold_results,
                    'fold_confidences': fold_confidences,
                    'best_fold': best_fold
                }
                np.save(f"preds/{tile_id}_all_folds_analysis_UTAE_semantic.npy", all_results)
            else:
                print(f"❌ Aucun résultat valide pour {tile_id}")

if __name__ == "__main__":
    run_inference_all_folds()