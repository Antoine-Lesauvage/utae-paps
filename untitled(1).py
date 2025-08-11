import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F
import cv2
from scipy.optimize import linear_sum_assignment
from datetime import datetime

with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

with open("NORM_S2_patch.json") as f:
    norm_vals = json.load(f)
os.environ["AWS_ACCESS_KEY_ID"] = 'ZDEMRP59PONSWI7XOOHA'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'vl4S7UvZp2N+MZ8AXVy6+nCb20QjvzQHrBBlgdFJ'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJaREVNUlA1OVBPTlNXSTdYT09IQSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0NDYyODUxLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1MDcwODA0LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDQ2NjAwNCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDplNzkwYjU1YS05NTZkLTlmOTgtMWYzNy03MTdiYzAzMjdjNzQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI2ZWQ5MzUwZi1hZjA0LTQxNzYtYTRkZi1lNzllZDNmM2YzMmMiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.zmYq-lXGCd628NwEnKcXMsUiXjq3C8cqJA-etTyK5x3fHCTwTSZNmksqKgmcg59b7l_1O1362s8Gv1dXEEUNCw'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1' 
# Convertir le dict en objet (similaire à un namespace)
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class S3SinglePatchDataset(Dataset):
    def __init__(self, s3_prefix, tile_ids, transform=None):
        self.s3_prefix = s3_prefix.rstrip("/")
        self.tile_ids = tile_ids
        self.transform = transform
        os.environ["AWS_ACCESS_KEY_ID"] = 'ZDEMRP59PONSWI7XOOHA'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'vl4S7UvZp2N+MZ8AXVy6+nCb20QjvzQHrBBlgdFJ'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJaREVNUlA1OVBPTlNXSTdYT09IQSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0NDYyODUxLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1MDcwODA0LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDQ2NjAwNCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDplNzkwYjU1YS05NTZkLTlmOTgtMWYzNy03MTdiYzAzMjdjNzQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI2ZWQ5MzUwZi1hZjA0LTQxNzYtYTRkZi1lNzllZDNmM2YzMmMiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.zmYq-lXGCd628NwEnKcXMsUiXjq3C8cqJA-etTyK5x3fHCTwTSZNmksqKgmcg59b7l_1O1362s8Gv1dXEEUNCw'
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


def load_model_and_predict(tile_id, fold_num, return_raw_output=True):
    """Charge le modèle et fait une prédiction pour récupérer les instances"""

    # Réutiliser votre pipeline existant mais retourner les outputs bruts
    s3_prefix = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS/DATA_S2"
    dataset = S3SinglePatchDataset(s3_prefix, [f"S2_{tile_id}"])
    dataloader = DataLoader(dataset, batch_size=1)
    
    model = load_model_fold(fold_num)
    
    for batch in dataloader:
        data = batch["data"][0].unsqueeze(0).cuda()
        
        # Normalisation
        mean = torch.tensor(norm_vals[f"Fold_{fold_num}"]["mean"]).float().cuda()
        std = torch.tensor(norm_vals[f"Fold_{fold_num}"]["std"]).float().cuda()
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
        x_norm = (data - mean) / std
        
        T, C, H, W = x_norm.shape[1:]
        batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = model(x_norm, batch_positions=batch_positions)
            
            if return_raw_output:
                return output, (H, W)
            
        break
    
    return None, None

def load_model_fold(fold_num):
    """Charge un modèle pour un fold spécifique"""
    model = model_utils.get_model(config, mode="panoptic")
    model.to(device)
    
    checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold_num}", "model.pth.tar")
    assert os.path.exists(checkpoint_path), f"Fichier introuvable : {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.cuda()
import cv2
from scipy.optimize import linear_sum_assignment

def load_pastis_annotations_for_evaluation(tile_id, pastis_root):
    """Charge les annotations panoptiques PASTIS pour une tuile"""
    
    # Chemins vers les fichiers d'annotation
    semantic_path = f"{pastis_root}/ANNOTATIONS/TARGET_{tile_id}.npy"
    instance_path = f"{pastis_root}/ANNOTATIONS/ParcelIDs_{tile_id}.npy"
    
    try:
        # Utiliser s3fs comme dans votre dataset
        fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
            key=os.environ["AWS_ACCESS_KEY_ID"], 
            secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
            token=os.environ["AWS_SESSION_TOKEN"]
        )
        
        # Charger les annotations sémantiques
        with fs.open(semantic_path, 'rb') as f:
            semantic_mask = np.load(f)
        
        # Charger les annotations d'instances
        with fs.open(instance_path, 'rb') as f:
            instance_mask = np.load(f)
        
        print(f"✅ Annotations chargées pour tuile {tile_id}")
        print(f"   Sémantique: {semantic_mask.shape}, classes: {np.unique(semantic_mask)}")
        print(f"   Instances: {instance_mask.shape}, instances: {len(np.unique(instance_mask))-1}")
        
        return {
            'semantic': semantic_mask,
            'instance': instance_mask,
            'tile_id': tile_id
        }
        
    except Exception as e:
        print(f"❌ Erreur chargement annotations {tile_id}: {e}")
        return None

def extract_ground_truth_instances(annotations, tile_id, pixel_size=10):
    """Extrait les instances ground truth des annotations PASTIS"""
    
    semantic_mask = annotations['semantic']
    instance_mask = annotations['instance']
    
    # Correction: prendre le premier canal du masque sémantique s'il est 3D
    if len(semantic_mask.shape) == 3:
        semantic_mask = semantic_mask[0]  # Prendre le premier canal (3,128,128) -> (128,128)
    
    # Obtenir les IDs d'instances uniques (exclure 0 = background)
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    gt_instances = []
    
    print(f"🎯 Extraction de {len(instance_ids)} instances GT...")
    print(f"   Semantic shape: {semantic_mask.shape}, Instance shape: {instance_mask.shape}")
    
    for i, instance_id in enumerate(instance_ids):
        try:
            # Masque binaire pour cette instance
            binary_mask = (instance_mask == instance_id).astype(np.uint8)
            
            # Classe sémantique majoritaire pour cette instance
            instance_pixels = semantic_mask[instance_mask == instance_id]
            unique_classes, counts = np.unique(instance_pixels, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            
            # Calculer les propriétés géométriques
            props = calculate_gt_instance_properties(
                binary_mask, tile_id, instance_id, majority_class, pixel_size
            )
            
            if props:
                gt_instances.append(props)
                
        except Exception as e:
            print(f"  ⚠️ Erreur instance GT {instance_id}: {e}")
            continue
    
    print(f"✅ {len(gt_instances)} instances GT extraites")
    return gt_instances
def calculate_gt_instance_properties(mask, tile_id, instance_id, class_id, pixel_size=10):
    """Calcule les propriétés d'une instance ground truth"""
    
    # Vérifier que le masque n'est pas vide
    if np.sum(mask) == 0:
        return None
    
    # Aire
    area_pixels = np.sum(mask)
    area_m2 = area_pixels * (pixel_size ** 2)
    area_ha = area_m2 / 10000
    
    # Filtrer les instances trop petites
    if area_pixels < 10:
        return None
    
    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Plus grand contour
    main_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(main_contour)
    
    if contour_area < 5:
        return None
    
    # Centre de masse
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        y_coords, x_coords = np.where(mask > 0)
        cx = int(np.mean(x_coords)) if len(x_coords) > 0 else 0
        cy = int(np.mean(y_coords)) if len(y_coords) > 0 else 0
    
    # Boîte englobante
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Coordonnées du polygone
    contour_points = main_contour.reshape(-1, 2).tolist()
    
    return {
        'tile_id': str(tile_id),
        'instance_id': f"GT_{tile_id}_{instance_id}",
        'class_id': int(class_id),
        'centroid_x': int(cx),
        'centroid_y': int(cy),
        'area_pixels': float(area_pixels),
        'area_m2': float(area_m2),
        'area_ha': float(area_ha),
        'bbox_x': int(x), 'bbox_y': int(y), 'bbox_w': int(w), 'bbox_h': int(h),
        'polygon_coords': contour_points,
        'source': 'ground_truth'
    }
def extract_instances_properties_corrected(data, tile_id, pixel_size=10):
    """Extrait les propriétés des instances (version corrigée pour les listes)"""
    
    instances_data = []
    
    # Convertir les tensors en numpy si nécessaire
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return x
    
    # Récupérer les composants
    instance_masks = data['instance_masks']  # Liste de masques
    semantic_logits = to_numpy(data['semantic'])        # (183, 20)
    confidence_scores = to_numpy(data['confidence'])    # (183, 1)
    instance_boxes = to_numpy(data['instance_boxes'])   # (183, 4)
    
    # Aplatir les scores de confiance si nécessaire
    if len(confidence_scores.shape) > 1:
        confidence_scores = confidence_scores.flatten()
    
    print(f"📊 Composants extraits:")
    print(f"  Instance masks: {len(instance_masks)} masques (liste)")
    print(f"  Semantic logits: {semantic_logits.shape}")
    print(f"  Confidence scores: {confidence_scores.shape}")
    print(f"  Instance boxes: {instance_boxes.shape}")
    
    # Vérifier la cohérence des tailles
    n_instances = len(instance_masks)
    if n_instances != semantic_logits.shape[0]:
        print(f"⚠️  Incohérence: {n_instances} masques vs {semantic_logits.shape[0]} prédictions sémantiques")
        n_instances = min(n_instances, semantic_logits.shape[0])
    
    # Calculer les classes prédites
    predicted_classes = np.argmax(semantic_logits, axis=1)
    # Calculer les probabilités (softmax)
    semantic_probs = np.exp(semantic_logits) / np.sum(np.exp(semantic_logits), axis=1, keepdims=True)
    class_probabilities = np.max(semantic_probs, axis=1)
    
    print(f"🏷️  Classes prédites: {np.unique(predicted_classes)}")
    print(f"📈 Confiances: min={confidence_scores.min():.3f}, max={confidence_scores.max():.3f}, mean={confidence_scores.mean():.3f}")
    print(f"🎯 Probabilités de classe: min={class_probabilities.min():.3f}, max={class_probabilities.max():.3f}")
    
    # Traiter chaque instance
    print(f"🔢 Traitement de {n_instances} instances...")
    
    for i in range(n_instances):
        try:
            # Récupérer le masque de l'instance
            mask = instance_masks[i]
            
            # Convertir le masque en numpy si c'est un tensor
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # Si le masque est 3D, prendre la première dimension
            if len(mask.shape) == 3:
                mask = mask[0]  # Prendre le premier canal
            
            class_id = predicted_classes[i]
            confidence = confidence_scores[i]
            class_prob = class_probabilities[i]
            bbox = instance_boxes[i]
            
            print(f"  Instance {i}: classe={class_id}, conf={confidence:.3f}, shape={mask.shape}")
            
            # Filtrer les instances avec une confiance trop faible
            if confidence < 0.2:  # Seuil ajustable
                print(f"    ❌ Confiance trop faible: {confidence:.3f}")
                continue
            
            # Extraire les propriétés géométriques
            props = calculate_instance_properties_corrected(
                mask, tile_id, i, class_id, confidence, class_prob, bbox, pixel_size
            )
            
            if props:
                instances_data.append(props)
                print(f"    ✅ Instance {i} extraite: {props['area_ha']:.4f} ha")
            else:
                print(f"    ❌ Instance {i} rejetée (propriétés invalides)")
                
        except Exception as e:
            print(f"  ⚠️ Erreur pour l'instance {i}: {e}")
            continue
    
    print(f"✅ {len(instances_data)} instances extraites avec succès")
    return instances_data

def calculate_instance_properties_corrected(mask, tile_id, instance_id, class_id, confidence, class_prob, bbox, pixel_size=10):
    """Version corrigée du calcul des propriétés"""
    
    # S'assurer que le masque est en numpy
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Convertir en masque binaire
    if mask.dtype != np.uint8:
        # Si c'est un masque de probabilité, seuiller
        if mask.max() <= 1.0 and mask.min() >= 0.0:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            # Si c'est déjà des valeurs entières
            binary_mask = (mask > 0).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)
    
    print(f"    Masque: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
    print(f"    Masque binaire: sum: {binary_mask.sum()}")
    
    # Vérifier que le masque n'est pas vide
    if np.sum(binary_mask) == 0:
        print(f"    ❌ Masque vide")
        return None
    
    # Aire en pixels
    area_pixels = np.sum(binary_mask)
    area_m2 = area_pixels * (pixel_size ** 2)
    area_ha = area_m2 / 10000
    
    # Filtrer les instances trop petites
    if area_pixels < 10:  # Minimum 10 pixels
        print(f"    ❌ Instance trop petite: {area_pixels} pixels")
        return None
    
    # Trouver les contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"    ❌ Aucun contour trouvé")
        return None
    
    # Prendre le plus grand contour
    main_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(main_contour)
    
    if contour_area < 5:
        print(f"    ❌ Contour trop petit: {contour_area}")
        return None
    
    # Périmètre
    perimeter_pixels = cv2.arcLength(main_contour, True)
    perimeter_m = perimeter_pixels * pixel_size
    
    # Centre de masse (centroïde)
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback: centre de masse du masque
        y_coords, x_coords = np.where(binary_mask > 0)
        cx = int(np.mean(x_coords)) if len(x_coords) > 0 else 0
        cy = int(np.mean(y_coords)) if len(y_coords) > 0 else 0
    
    # Boîte englobante du contour
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Boîte englobante prédite par le modèle
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    bbox_w_pred = bbox_x2 - bbox_x1
    bbox_h_pred = bbox_y2 - bbox_y1
    
    # Ellipse ajustée
    if len(main_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(main_contour)
            ellipse_center = ellipse[0]
            ellipse_axes = ellipse[1]
            ellipse_angle = ellipse[2]
        except:
            ellipse_center = (cx, cy)
            ellipse_axes = (w, h)
            ellipse_angle = 0
    else:
        ellipse_center = (cx, cy)
        ellipse_axes = (w, h)
        ellipse_angle = 0
    
    # Indicateurs de forme
    compactness = (4 * np.pi * contour_area) / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
    
    # Convexité
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    convexity = contour_area / hull_area if hull_area > 0 else 0
    
    # Solidité
    solidity = contour_area / hull_area if hull_area > 0 else 0
    
    # Extent
    extent = contour_area / (w * h) if (w * h) > 0 else 0
    
    # Coordonnées du polygone
    contour_points = main_contour.reshape(-1, 2).tolist()
    
    # Diamètre équivalent
    equivalent_diameter = np.sqrt(4 * contour_area / np.pi)
    
    # Retourner avec types JSON-safe
    return {
        'tile_id': str(tile_id),
        'instance_id': f"{tile_id}_{instance_id}",
        'class_id': int(class_id),
        'confidence': float(confidence),
        'class_probability': float(class_prob),
        
        # Centre
        'centroid_x': int(cx),
        'centroid_y': int(cy),
        'centroid_x_m': float(cx * pixel_size),
        'centroid_y_m': float(cy * pixel_size),
        
        # Surface
        'area_pixels': float(area_pixels),
        'area_m2': float(area_m2),
        'area_ha': float(area_ha),
        'contour_area_pixels': float(contour_area),
        
        # Périmètre
        'perimeter_pixels': float(perimeter_pixels),
        'perimeter_m': float(perimeter_m),
        
        # Boîte englobante (contour réel)
        'bbox_x': int(x), 'bbox_y': int(y), 'bbox_w': int(w), 'bbox_h': int(h),
        'bbox_width_m': float(w * pixel_size),
        'bbox_height_m': float(h * pixel_size),
        
        # Boîte englobante prédite
        'predicted_bbox_x1': float(bbox_x1),
        'predicted_bbox_y1': float(bbox_y1),
        'predicted_bbox_x2': float(bbox_x2),
        'predicted_bbox_y2': float(bbox_y2),
        'predicted_bbox_width_m': float(bbox_w_pred * pixel_size),
        'predicted_bbox_height_m': float(bbox_h_pred * pixel_size),
        
        # Ellipse
        'ellipse_center_x': float(ellipse_center[0]),
        'ellipse_center_y': float(ellipse_center[1]),
        'ellipse_major_axis_m': float(max(ellipse_axes) * pixel_size),
        'ellipse_minor_axis_m': float(min(ellipse_axes) * pixel_size),
        'ellipse_angle': float(ellipse_angle),
        
        # Indicateurs de forme
        'compactness': float(compactness),
        'aspect_ratio': float(aspect_ratio),
        'convexity': float(convexity),
        'solidity': float(solidity),
        'extent': float(extent),
        'equivalent_diameter_m': float(equivalent_diameter * pixel_size),
        
        # Géométrie
        'polygon_coords': [[int(pt[0]), int(pt[1])] for pt in contour_points],
        'n_vertices': int(len(contour_points))
    }
def run_evaluation_on_training_tile(tile_id, pastis_root, confidence_threshold=0.3):
    """Évalue tous les folds sur une tuile d'entraînement"""
    
    print(f"🎯 ÉVALUATION SUR DONNÉES D'ENTRAÎNEMENT - Tuile {tile_id}")
    print("=" * 60)
    
    # 1. Charger les annotations ground truth
    annotations = load_pastis_annotations_for_evaluation(tile_id, pastis_root)
    if not annotations:
        print("❌ Impossible de charger les annotations")
        return None
    
    gt_instances = extract_ground_truth_instances(annotations, tile_id)
    if not gt_instances:
        print("❌ Aucune instance GT extraite")
        return None
    
    print(f"🎯 Ground Truth: {len(gt_instances)} parcelles")
    
    # 2. Inférence sur tous les folds
    fold_results = {}
    
    for fold_num in range(1, 6):
        print(f"\n🔄 Évaluation Fold {fold_num}")
        print("-" * 30)
        
        try:
            # Faire l'inférence
            output, image_shape = load_model_and_predict(tile_id, fold_num, return_raw_output=True)
            
            if output is None:
                print(f"  ❌ Échec de l'inférence")
                continue
            
            # Extraire les instances prédites
            pred_instances = extract_instances_properties_corrected(output, tile_id)
            
            # Filtrer par confiance
            pred_instances_filtered = [
                inst for inst in pred_instances 
                if inst['confidence'] >= confidence_threshold
            ]
            
            print(f"  📊 Prédictions: {len(pred_instances)} total, {len(pred_instances_filtered)} après filtrage")
            
            # Évaluer contre GT
            if pred_instances_filtered:
                metrics = evaluate_predictions_vs_ground_truth(
                    pred_instances_filtered, gt_instances, tile_id, fold_num
                )
                fold_results[fold_num] = {
                    'predictions': pred_instances_filtered,
                    'metrics': metrics,
                    'n_predictions': len(pred_instances_filtered)
                }
            else:
                fold_results[fold_num] = {
                    'predictions': [],
                    'metrics': None,
                    'n_predictions': 0
                }
                
        except Exception as e:
            print(f"  ❌ Erreur fold {fold_num}: {e}")
            fold_results[fold_num] = {'error': str(e)}
    
    # 3. Résumé comparatif
    comparison_results = {
        'tile_id': tile_id,
        'ground_truth': gt_instances,
        'fold_results': fold_results,
        'n_gt_instances': len(gt_instances)
    }
    
    print_fold_comparison(fold_results, tile_id)
    
    return comparison_results

def evaluate_predictions_vs_ground_truth(pred_instances, gt_instances, tile_id, fold_num):
    """Évalue les prédictions contre le ground truth"""
    
    print(f"  📊 Évaluation Fold {fold_num}:")
    print(f"    Prédictions: {len(pred_instances)}")
    print(f"    Ground Truth: {len(gt_instances)}")
    
    if len(pred_instances) == 0 or len(gt_instances) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'mean_iou': 0.0, 'classification_accuracy': 0.0,
            'n_matches': 0, 'n_predictions': len(pred_instances), 'n_ground_truth': len(gt_instances)
        }
    
    # 1. Calculer les IoU entre toutes les paires
    iou_matrix = compute_iou_matrix_instances(pred_instances, gt_instances)
    
    # 2. Assignment optimal (Hungarian algorithm)
    cost_matrix = 1 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # 3. Filtrer les assignments avec IoU suffisant
    iou_threshold = 0.3
    valid_matches = []
    
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[p_idx, g_idx]
        if iou >= iou_threshold:
            pred_class = pred_instances[p_idx]['class_id']
            gt_class = gt_instances[g_idx]['class_id']
            
            match = {
                'pred_idx': p_idx,
                'gt_idx': g_idx,
                'iou': iou,
                'pred_class': pred_class,
                'gt_class': gt_class,
                'class_correct': (pred_class == gt_class),
                'confidence': pred_instances[p_idx]['confidence']
            }
            valid_matches.append(match)
    
    # 4. Calculer les métriques
    n_matches = len(valid_matches)
    precision = n_matches / len(pred_instances) if len(pred_instances) > 0 else 0
    recall = n_matches / len(gt_instances) if len(gt_instances) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU moyen des matches
    mean_iou = np.mean([m['iou'] for m in valid_matches]) if valid_matches else 0.0
    
    # Accuracy de classification sur les matches
    correct_classifications = sum(1 for m in valid_matches if m['class_correct'])
    classification_accuracy = correct_classifications / n_matches if n_matches > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'classification_accuracy': classification_accuracy,
        'n_matches': n_matches,
        'n_predictions': len(pred_instances),
        'n_ground_truth': len(gt_instances),
        'matches': valid_matches
    }
    
    print(f"    📈 Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
    print(f"    📐 IoU moyen: {mean_iou:.3f}, Acc. classification: {classification_accuracy:.3f}")
    
    return metrics

def compute_iou_matrix_instances(pred_instances, gt_instances):
    """Calcule la matrice IoU entre prédictions et ground truth"""
    
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred in enumerate(pred_instances):
        pred_coords = np.array(pred['polygon_coords'])
        
        for j, gt in enumerate(gt_instances):
            gt_coords = np.array(gt['polygon_coords'])
            
            # Calculer IoU avec Shapely
            iou = compute_polygon_iou_shapely(pred_coords, gt_coords)
            iou_matrix[i, j] = iou
    
    return iou_matrix

def compute_polygon_iou_shapely(poly1_coords, poly2_coords):
    """Calcule l'IoU entre deux polygones avec Shapely"""
    
    try:
        from shapely.geometry import Polygon
        
        if len(poly1_coords) < 3 or len(poly2_coords) < 3:
            return 0.0
        
        p1 = Polygon(poly1_coords)
        p2 = Polygon(poly2_coords)
        
        # Nettoyer les polygones si nécessaire
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        
        if p1.is_valid and p2.is_valid and not p1.is_empty and not p2.is_empty:
            intersection = p1.intersection(p2)
            union = p1.union(p2)
            
            if union.area > 0:
                return intersection.area / union.area
        
        return 0.0
        
    except Exception:
        return 0.0
def print_fold_comparison(fold_results, tile_id):
    """Affiche la comparaison entre les folds"""
    
    print(f"\n📊 COMPARAISON DES FOLDS - Tuile {tile_id}")
    print("=" * 60)
    
    # Tableau comparatif
    print("Fold | Pred | F1-Score | Precision | Recall | IoU moy | Class Acc | Matches")
    print("-" * 75)
    
    fold_scores = {}
    
    for fold_num in range(1, 6):
        if fold_num in fold_results and 'metrics' in fold_results[fold_num]:
            if fold_results[fold_num]['metrics']:
                m = fold_results[fold_num]['metrics']
                n_pred = m['n_predictions']
                f1 = m['f1_score']
                precision = m['precision']
                recall = m['recall']
                iou = m['mean_iou']
                acc = m['classification_accuracy']
                matches = m['n_matches']
                
                print(f" {fold_num}   | {n_pred:4d} | {f1:8.3f} | {precision:9.3f} | {recall:6.3f} | {iou:7.3f} | {acc:9.3f} | {matches:7d}")
                
                # Score global combiné
                global_score = (f1 + iou + acc) / 3
                fold_scores[fold_num] = global_score
            else:
                print(f" {fold_num}   |    0 |    0.000 |     0.000 |  0.000 |   0.000 |     0.000 |       0")
        else:
            print(f" {fold_num}   | ERR  |      ERR |       ERR |    ERR |     ERR |       ERR |     ERR")
    
    # Meilleur fold
    if fold_scores:
        best_fold = max(fold_scores, key=fold_scores.get)
        print(f"\n🏆 MEILLEUR FOLD: Fold {best_fold} (score global: {fold_scores[best_fold]:.3f})")
        
        return best_fold, fold_scores
    
    return None, {}

def run_evaluation_on_multiple_tiles(tile_ids, pastis_root):
    """Évalue sur plusieurs tuiles d'entraînement"""
    
    print(f"🎯 ÉVALUATION SUR {len(tile_ids)} TUILES D'ENTRAÎNEMENT")
    print("=" * 60)
    
    all_results = {}
    fold_performance_summary = {i: [] for i in range(1, 6)}
    
    for tile_id in tile_ids:
        print(f"\n{'='*80}")
        results = run_evaluation_on_training_tile(tile_id, pastis_root)
        
        if results:
            all_results[tile_id] = results
            
            # Collecter les scores de performance
            for fold_num in range(1, 6):
                if fold_num in results['fold_results'] and 'metrics' in results['fold_results'][fold_num]:
                    metrics = results['fold_results'][fold_num]['metrics']
                    if metrics:
                        global_score = (metrics['f1_score'] + metrics['mean_iou'] + metrics['classification_accuracy']) / 3
                        fold_performance_summary[fold_num].append(global_score)
                    else:
                        fold_performance_summary[fold_num].append(0.0)
    
    # Analyse globale
    print(f"\n🏆 ANALYSE GLOBALE SUR {len(tile_ids)} TUILES")
    print("=" * 60)
    
    print("Fold | Moy Score | Médiane | Std Dev | Min | Max | Nb Succès")
    print("-" * 65)
    
    fold_rankings = {}
    
    for fold_num in range(1, 6):
        scores = fold_performance_summary[fold_num]
        if scores:
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            success_rate = sum(1 for s in scores if s > 0.1) / len(scores)  # Seuil arbitraire
            
            print(f" {fold_num}   | {mean_score:9.3f} | {median_score:7.3f} | {std_score:7.3f} | {min_score:.3f} | {max_score:.3f} | {success_rate:9.1%}")
            fold_rankings[fold_num] = mean_score
        else:
            print(f" {fold_num}   |     0.000 |   0.000 |   0.000 | 0.000 | 0.000 |     0.0%")
    
    # Recommandation finale
    if fold_rankings:
        best_overall_fold = max(fold_rankings, key=fold_rankings.get)
        print(f"\n🎉 RECOMMANDATION: Utilisez le Fold {best_overall_fold}")
        print(f"   Score moyen: {fold_rankings[best_overall_fold]:.3f}")
        print(f"   Écart avec le 2e meilleur: {fold_rankings[best_overall_fold] - sorted(fold_rankings.values())[-2]:.3f}")
    
    # Sauvegarder les résultats
    with open('fold_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results, fold_rankings
def analyze_fold_performance_detailed(results, save_plots=True):
    """Analyse détaillée des performances par fold"""
    
    print("\n📊 ANALYSE DÉTAILLÉE DES PERFORMANCES")
    print("=" * 60)
    
    # Collecter toutes les métriques
    fold_metrics = {i: {
        'f1_scores': [], 'precisions': [], 'recalls': [], 
        'ious': [], 'class_accuracies': []
    } for i in range(1, 6)}
    
    for tile_id, tile_results in results.items():
        for fold_num, fold_data in tile_results['fold_results'].items():
            if 'metrics' in fold_data and fold_data['metrics']:
                m = fold_data['metrics']
                fold_metrics[fold_num]['f1_scores'].append(m['f1_score'])
                fold_metrics[fold_num]['precisions'].append(m['precision'])
                fold_metrics[fold_num]['recalls'].append(m['recall'])
                fold_metrics[fold_num]['ious'].append(m['mean_iou'])
                fold_metrics[fold_num]['class_accuracies'].append(m['classification_accuracy'])
    
    # Statistiques par fold
    for fold_num in range(1, 6):
        print(f"\n📈 Fold {fold_num}:")
        metrics = fold_metrics[fold_num]
        
        if metrics['f1_scores']:
            print(f"  F1-Score    : {np.mean(metrics['f1_scores']):.3f} ± {np.std(metrics['f1_scores']):.3f}")
            print(f"  Precision   : {np.mean(metrics['precisions']):.3f} ± {np.std(metrics['precisions']):.3f}")
            print(f"  Recall      : {np.mean(metrics['recalls']):.3f} ± {np.std(metrics['recalls']):.3f}")
            print(f"  IoU moyen   : {np.mean(metrics['ious']):.3f} ± {np.std(metrics['ious']):.3f}")
            print(f"  Class Acc   : {np.mean(metrics['class_accuracies']):.3f} ± {np.std(metrics['class_accuracies']):.3f}")
        else:
            print("  Aucune donnée disponible")
    
    return fold_metrics

def save_evaluation_report(results, rankings, output_file="fold_evaluation_report.json"):
    """Sauvegarde un rapport complet d'évaluation"""
    
    report = {
        'evaluation_timestamp': str(datetime.now()),
        'summary': {
            'n_tiles_evaluated': len(results),
            'fold_rankings': rankings,
            'best_fold': max(rankings, key=rankings.get) if rankings else None
        },
        'detailed_results': results,
        'recommendations': {
            'best_fold': max(rankings, key=rankings.get) if rankings else None,
            'confidence_level': 'high' if len(results) >= 3 else 'medium'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Rapport sauvegardé: {output_file}")
    return report
# Script principal pour l'évaluation des folds
def main_fold_evaluation():
    """Script principal d'évaluation des folds (version corrigée)"""
    
    print("🚀 ÉVALUATION DES FOLDS SUR DONNÉES D'ENTRAÎNEMENT")
    print("=" * 80)
    
    # Configuration
    training_tiles = ["10000"]  # Tuiles de test
    pastis_root = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS"
    
    # Vérifier d'abord la disponibilité des annotations
    print("🔍 Vérification des annotations...")
    for tile_id in training_tiles:
        annotations = load_pastis_annotations_for_evaluation(tile_id, pastis_root)
        if not annotations:
            print(f"❌ Annotations manquantes pour tuile {tile_id}")
            return None, None
        else:
            print(f"✅ Annotations OK pour tuile {tile_id}")
    
    # Test sur une tuile d'abord
    print("\n📊 Phase 1: Test sur une tuile")
    test_result = run_evaluation_on_training_tile(training_tiles[0], pastis_root)
    
    if test_result:
        print("✅ Test réussi!")
        
        # Analyse des résultats du test
        best_fold, fold_scores = print_fold_comparison(
            test_result['fold_results'], 
            training_tiles[0]
        )
        
        if best_fold:
            print(f"\n🏆 Pour cette tuile, le meilleur fold est: {best_fold}")
        
        # Étendre à d'autres tuiles si souhaité
        extended_tiles = ["10000", "10001", "10002"]  # Ajustez selon vos données
        
        if len(extended_tiles) > 1:
            print(f"\n📈 Phase 2: Extension à {len(extended_tiles)} tuiles")
            results, rankings = run_evaluation_on_multiple_tiles(extended_tiles, pastis_root)
            return results, rankings
        else:
            return {training_tiles[0]: test_result}, fold_scores
    else:
        print("❌ Échec du test initial")
        return None, None




        
# Puis lancez l'évaluation
if __name__ == "__main__":
    results, rankings = main_fold_evaluation()
    
    if results and rankings:
        # Analyse détaillée
        fold_metrics = analyze_fold_performance_detailed(results)
        
        # Sauvegarde du rapport
        report = save_evaluation_report(results, rankings)
        
        print(f"\n🎉 ÉVALUATION TERMINÉE!")
        print(f"🏆 Meilleur fold recommandé: {report['recommendations']['best_fold']}")
    else:
        print("❌ Évaluation échouée")