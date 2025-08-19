import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F
import logging
import time
from datetime import datetime
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fold = 2
with open("/home/onyxia/work/UATE_zenodo/conf.json") as f:
    config_dict = json.load(f)

with open("NORM_S2_patch.json") as f:
    norm_vals = json.load(f)
 
config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class S3FileManager:
    """Gestionnaire S3 thread-safe"""
    def __init__(self):
        os.environ["AWS_ACCESS_KEY_ID"] = 'VWDWXJFA2NU5TE5U03ER'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'fmTH5ymD+gcjkc+6qgrXNq+OUNTTkxj3UsyMsnNf'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJWV0RXWEpGQTJOVTVURTVVMDNFUiIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU1MDY4NDc2LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1Njc0MDk0LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NTA2OTI5NCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDo0ZWI5NWJkOC0wN2YzLTVkZTYtNmRhMy00NTc2Njc3MGQyMzgiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI4MTJjNzE3OC03YmQ5LTRjYzQtOTUxNy0yNzAyNDdhODhkODgiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.C1F-8ZPGP2jHSWgHFiAYnuhAtqPty9-2hBKyopHcnfAKDWbVg818VtWKPyRvSsGyJC8kOZ9gnsaYWybHtgUDsg'
        os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
        
        # Thread-local storage pour les connexions S3
        self._local = threading.local()
    
    @property
    def fs(self):
        """Connexion S3 thread-safe"""
        if not hasattr(self._local, 'fs'):
            self._local.fs = s3fs.S3FileSystem(
                client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
                key=os.environ["AWS_ACCESS_KEY_ID"], 
                secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
                token=os.environ["AWS_SESSION_TOKEN"]
            )
        return self._local.fs
    
    def file_exists(self, s3_path):
        try:
            return self.fs.exists(s3_path)
        except:
            return False
    
    def load_numpy(self, s3_path):
        """Charge un array numpy depuis S3"""
        try:
            with self.fs.open(s3_path, 'rb') as f:
                return np.load(f)
        except Exception as e:
            raise Exception(f"Erreur chargement {s3_path}: {e}")
    
    def save_numpy(self, data, s3_path):
        try:
            buffer = io.BytesIO()
            np.save(buffer, data, allow_pickle=True)
            buffer.seek(0)
            
            with self.fs.open(s3_path, 'wb') as f:
                f.write(buffer.getvalue())
            return True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde S3 {s3_path}: {e}")
            return False
    
    def save_json(self, data, s3_path):
        """Sauvegarde un JSON sur S3"""
        try:
            with self.fs.open(s3_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde JSON S3 {s3_path}: {e}")
            return False

class BatchInferenceDataset(Dataset):
    """Dataset optimisé pour le traitement par lots"""
    def __init__(self, tile_ids, s3_input_prefix, s3_manager):
        self.tile_ids = tile_ids
        self.s3_input_prefix = s3_input_prefix.rstrip("/")
        self.s3_manager = s3_manager

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        try:
            path = f"{self.s3_input_prefix}/{tile_id}.npy"
            arr = self.s3_manager.load_numpy(path)
            tensor = torch.from_numpy(arr).float()
            return {"tile_id": tile_id, "data": tensor, "success": True}
        except Exception as e:
            # Retourner un placeholder en cas d'erreur
            return {"tile_id": tile_id, "data": None, "success": False, "error": str(e)}

def get_all_tile_ids(s3_prefix, s3_manager):
    """Récupère tous les tile_ids depuis S3"""
    try:
        files = s3_manager.fs.glob(f"{s3_prefix.rstrip('/')}/*.npy")
        tile_ids = [file_path.split('/')[-1].replace('.npy', '') for file_path in files]
        logger.info(f"🎯 Trouvé {len(tile_ids)} tuiles dans {s3_prefix}")
        return sorted(tile_ids)
    except Exception as e:
        logger.error(f"❌ Erreur listage: {e}")
        return []

def load_model_fold(fold_num):
    """Charge un modèle UTAE sémantique"""
    model = model_utils.get_model(config, mode="semantic")
    model.to(device)
    
    checkpoint_path = os.path.join("/home/onyxia/work/UATE_zenodo/", f"Fold_{fold_num}", "model.pth.tar")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not any(exclude in k for exclude in ['paps', 'panoptic', 'instance'])}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model

def calculate_confidence_utae(output):
    """Calcule la confiance pour UTAE"""
    if torch.is_tensor(output):
        semantic_logits = output
    else:
        semantic_logits = None
        for key in ["semantic", "out", "logits", "pred"]:
            if key in output:
                semantic_logits = output[key]
                break
    
    if semantic_logits is None or not torch.is_tensor(semantic_logits):
        return 0.0
    
    if len(semantic_logits.shape) == 4:
        dim = 1
    elif len(semantic_logits.shape) == 3:
        dim = 0
    else:
        return 0.0
    
    probs = F.softmax(semantic_logits, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
    max_entropy = torch.log(torch.tensor(float(probs.shape[dim]), device=probs.device))
    confidence = 1 - (entropy / max_entropy)
    return confidence.mean().item()

def process_utae_output(output):
    """Traite les sorties UTAE"""
    if torch.is_tensor(output):
        semantic_logits = output
    else:
        semantic_logits = None
        for key in ["semantic", "out", "logits", "pred"]:
            if key in output:
                semantic_logits = output[key]
                break
    
    if semantic_logits is None:
        return None, None
    
    if len(semantic_logits.shape) == 4:  # (B, C, H, W)
        semantic_pred = semantic_logits.argmax(dim=1)[0].cpu().numpy()
        probs = F.softmax(semantic_logits, dim=1)[0].cpu().numpy()
        max_probs = np.max(probs, axis=0)
    elif len(semantic_logits.shape) == 3:  # (C, H, W)
        semantic_pred = semantic_logits.argmax(dim=0).cpu().numpy()
        probs = F.softmax(semantic_logits, dim=0).cpu().numpy()
        max_probs = np.max(probs, axis=0)
    else:
        return None, None
    
    return semantic_pred, max_probs

# ✅ NOUVELLES FONCTIONS CORRIGÉES
def save_batch_results(tile_ids, all_fold_results, all_fold_confidences, s3_output_prefix, s3_manager, batch_id):
    """Sauvegarde les résultats d'un batch"""
    results = []
    
    for tile_id in tile_ids:
        if tile_id in all_fold_confidences and all_fold_confidences[tile_id]:
            best_fold = max(all_fold_confidences[tile_id], key=all_fold_confidences[tile_id].get)
            best_confidence = all_fold_confidences[tile_id][best_fold]
            
            # Chemins de sauvegarde
            best_result_path = f"{s3_output_prefix.rstrip('/')}/{tile_id}_best_fold_{best_fold}_UTAE_semantic.npy"
            #analysis_path = f"{s3_output_prefix.rstrip('/')}/{tile_id}_all_folds_analysis_UTAE_semantic.npy"
            
            # Vérifier si déjà existant
            if s3_manager.file_exists(best_result_path):
                results.append(f"⏭️ {tile_id}: Déjà traité")
                continue
            
            # Préparer les données
            best_result = all_fold_results[tile_id][best_fold]
            #analysis_result = {
            #    'fold_results': all_fold_results[tile_id],
            #    'fold_confidences': all_fold_confidences[tile_id],
            #    'best_fold': best_fold,
            #    'batch_id': batch_id,
            #    'timestamp': datetime.now().isoformat()
            #}
            
            # Sauvegarder
            if s3_manager.save_numpy(best_result, best_result_path):
                results.append(f"✅ {tile_id}: Fold {best_fold} (conf: {best_confidence:.4f})")
            else:
                results.append(f"❌ {tile_id}: Erreur sauvegarde S3")
        else:
            results.append(f"❌ {tile_id}: Aucun résultat valide")
    
    return results

def process_uniform_batch(tile_ids, tensors, models, s3_output_prefix, s3_manager, batch_id):
    """Traite un lot de tuiles avec dimensions uniformes"""
    results = []
    
    # Empiler directement (toutes les dimensions sont identiques)
    batch_tensor = torch.stack(tensors).cuda()  # (B, T, C, H, W)
    
    # Traitement avec tous les folds
    all_fold_results = {}
    all_fold_confidences = {}
    
    with torch.no_grad():
        for fold in range(1, 6):
            # Normalisation
            mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
            std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda()
            mean = mean.reshape(1, 1, -1, 1, 1)
            std = std.reshape(1, 1, -1, 1, 1)
            
            x_norm = (batch_tensor - mean) / std
            T = x_norm.shape[1]
            batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).repeat(len(tile_ids), 1).cuda()
            
            # Inférence du fold
            fold_outputs = models[fold](x_norm, batch_positions=batch_positions)
            
            # Traiter chaque sortie du batch
            for i, tile_id in enumerate(tile_ids):
                if torch.is_tensor(fold_outputs):
                    tile_output = fold_outputs[i:i+1]
                else:
                    tile_output = {k: v[i:i+1] if torch.is_tensor(v) else v for k, v in fold_outputs.items()}
                
                # Calculer confiance et prédiction
                confidence = calculate_confidence_utae(tile_output)
                semantic_pred, max_probs = process_utae_output(tile_output)
                
                if tile_id not in all_fold_results:
                    all_fold_results[tile_id] = {}
                    all_fold_confidences[tile_id] = {}
                
                if semantic_pred is not None:
                    all_fold_results[tile_id][fold] = {
                        'semantic_pred': semantic_pred,
                        'confidence_map': max_probs,
                        'raw_output': tile_output.cpu().numpy() if torch.is_tensor(tile_output) else 
                                     {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tile_output.items()}
                    }
                    all_fold_confidences[tile_id][fold] = confidence
    
    # Sauvegarder les résultats
    results.extend(save_batch_results(tile_ids, all_fold_results, all_fold_confidences, s3_output_prefix, s3_manager, batch_id))
    
    # Nettoyage
    del batch_tensor, x_norm
    torch.cuda.empty_cache()
    
    return results

def process_mixed_batch(tile_ids, tensors, models, s3_output_prefix, s3_manager, batch_id):
    """Traite un lot de tuiles avec dimensions mixtes (une par une)"""
    results = []
    
    all_fold_results = {}
    all_fold_confidences = {}
    
    with torch.no_grad():
        for fold in range(1, 6):
            # Normalisation
            mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
            std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda()
            mean = mean.reshape(1, 1, -1, 1, 1)
            std = std.reshape(1, 1, -1, 1, 1)
            
            for i, (tile_id, tensor) in enumerate(zip(tile_ids, tensors)):
                # Traiter une tuile à la fois
                x = tensor.unsqueeze(0).cuda()  # (1, T, C, H, W)
                x_norm = (x - mean) / std
                
                T = x_norm.shape[1]
                batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()  # (1, T)
                
                # Inférence
                tile_output = models[fold](x_norm, batch_positions=batch_positions)
                
                # Calculer confiance et prédiction
                confidence = calculate_confidence_utae(tile_output)
                semantic_pred, max_probs = process_utae_output(tile_output)
                
                if tile_id not in all_fold_results:
                    all_fold_results[tile_id] = {}
                    all_fold_confidences[tile_id] = {}
                
                if semantic_pred is not None:
                    all_fold_results[tile_id][fold] = {
                        'semantic_pred': semantic_pred,
                        'confidence_map': max_probs,
                        'raw_output': tile_output.cpu().numpy() if torch.is_tensor(tile_output) else 
                                     {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tile_output.items()}
                    }
                    all_fold_confidences[tile_id][fold] = confidence
                
                # Nettoyage après chaque tuile
                del x, x_norm, tile_output
            
            # Nettoyage après chaque fold
            torch.cuda.empty_cache()
    
    # Sauvegarder les résultats
    results.extend(save_batch_results(tile_ids, all_fold_results, all_fold_confidences, s3_output_prefix, s3_manager, batch_id))
    
    return results

def process_individual_tiles(tile_ids, s3_input_prefix, models, s3_output_prefix, s3_manager, start_idx=0):
    """Traite les tuiles une par une - AUCUN batch processing"""
    
    results = []
    
    for i, tile_id in enumerate(tile_ids):
        tile_start_time = time.time()
        current_idx = start_idx + i
        
        try:
            # Chemins de sauvegarde
            best_result_path = f"{s3_output_prefix.rstrip('/')}/{tile_id}_best_fold_UTAE_semantic.npy"
            #analysis_path = f"{s3_output_prefix.rstrip('/')}/{tile_id}_all_folds_analysis_UTAE_semantic.npy"
            
            # Vérifier si déjà traité
            if s3_manager.file_exists(best_result_path):
                logger.info(f"⏭️  [{current_idx+1}] {tile_id} - Déjà traité")
                results.append(f"⏭️ {tile_id}: Déjà traité")
                continue
            
            # Charger la tuile
            tile_path = f"{s3_input_prefix.rstrip('/')}/{tile_id}.npy"
            try:
                tensor = torch.from_numpy(s3_manager.load_numpy(tile_path)).float()
            except Exception as e:
                logger.error(f"❌ [{current_idx+1}] {tile_id} - Erreur chargement: {e}")
                results.append(f"❌ {tile_id}: Erreur chargement - {str(e)}")
                continue
            
            logger.info(f"🔍 [{current_idx+1}/{len(tile_ids)}] Traitement {tile_id} - Shape: {tensor.shape}")
            
            # Traiter tous les folds pour cette tuile
            fold_results = {}
            fold_confidences = {}
            
            with torch.no_grad():
                for fold in range(1, 6):
                    try:
                        # Normalisation du fold
                        mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().cuda()
                        std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().cuda()
                        mean = mean.reshape(1, 1, -1, 1, 1)
                        std = std.reshape(1, 1, -1, 1, 1)
                        
                        # Préparer la tuile pour l'inférence
                        x = tensor.unsqueeze(0).cuda()  # (1, T, C, H, W)
                        x_norm = (x - mean) / std
                        
                        T = x_norm.shape[1]
                        batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).cuda()  # (1, T)
                        
                        # Inférence
                        tile_output = models[fold](x_norm, batch_positions=batch_positions)
                        
                        # Calculer confiance et prédiction
                        confidence = calculate_confidence_utae(tile_output)
                        semantic_pred, max_probs = process_utae_output(tile_output)
                        
                        if semantic_pred is not None:
                            fold_results[fold] = {
                                'semantic_pred': semantic_pred,
                                'confidence_map': max_probs,
                                'raw_output': tile_output.cpu().numpy() if torch.is_tensor(tile_output) else 
                                             {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tile_output.items()}
                            }
                            fold_confidences[fold] = confidence
                        
                        # Nettoyage immédiat
                        del x, x_norm, tile_output
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.warning(f"  ⚠️  Fold {fold} échoué pour {tile_id}: {e}")
                        continue
            
            # Sélectionner le meilleur fold
            if fold_confidences:
                best_fold = max(fold_confidences, key=fold_confidences.get)
                best_confidence = fold_confidences[best_fold]
                
                # Préparer les données à sauvegarder
                best_result = fold_results[best_fold]
                #analysis_result = {
                #    'fold_results': fold_results,
                #    'fold_confidences': fold_confidences,
                #    'best_fold': best_fold,
                #    'processing_time': time.time() - tile_start_time,
                #    'timestamp': datetime.now().isoformat()
                #}
                
                # Sauvegarder
                if s3_manager.save_numpy(best_result, best_result_path):
                    
                    processing_time = time.time() - tile_start_time
                    logger.info(f"✅ [{current_idx+1}] {tile_id} - Fold {best_fold} (conf: {best_confidence:.4f}) - {processing_time:.1f}s")
                    results.append(f"✅ {tile_id}: Fold {best_fold} (confiance: {best_confidence:.4f})")
                else:
                    logger.error(f"❌ [{current_idx+1}] {tile_id} - Erreur sauvegarde S3")
                    results.append(f"❌ {tile_id}: Erreur sauvegarde S3")
            else:
                logger.error(f"❌ [{current_idx+1}] {tile_id} - Aucun résultat valide")
                results.append(f"❌ {tile_id}: Aucun résultat valide")
            
        except Exception as e:
            logger.error(f"❌ [{current_idx+1}] {tile_id} - Erreur générale: {e}")
            results.append(f"❌ {tile_id}: Erreur - {str(e)}")
    
    return results

def run_individual_inference(s3_input_prefix, s3_output_prefix, chunk_size=100, max_workers=2):
    """Inférence individuelle avec parallélisation par chunks"""
    
    logger.info(f"""
🚀 INFÉRENCE INDIVIDUELLE UTAE (14k+ tuiles)
════════════════════════════════════════════════════════════════════════════════
📥 Input: {s3_input_prefix}
📤 Output: {s3_output_prefix}
📦 Chunk size: {chunk_size}
👥 Workers: {max_workers}
🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
════════════════════════════════════════════════════════════════════════════════
    """)
    
    start_time = time.time()
    s3_manager = S3FileManager()
    
    # Récupérer toutes les tuiles
    all_tile_ids = get_all_tile_ids(s3_input_prefix, s3_manager)
    if not all_tile_ids:
        logger.error("❌ Aucune tuile trouvée!")
        return []
    
    logger.info(f"🎯 {len(all_tile_ids)} tuiles à traiter")
    
    # Charger tous les modèles une seule fois
    logger.info("📦 Chargement des modèles...")
    models = {}
    for fold in range(1, 6):
        models[fold] = load_model_fold(fold)
        logger.info(f"  ✅ Modèle Fold_{fold} chargé")
    
    # Diviser en chunks pour le parallélisme
    chunks = [all_tile_ids[i:i+chunk_size] for i in range(0, len(all_tile_ids), chunk_size)]
    logger.info(f"📦 {len(chunks)} chunks de {chunk_size} tuiles max")
    
    all_results = []
    processed_chunks = 0
    
    def process_chunk(chunk_tile_ids, chunk_id):
        """Traite un chunk de tuiles"""
        start_idx = chunk_id * chunk_size
        return process_individual_tiles(chunk_tile_ids, s3_input_prefix, models, s3_output_prefix, s3_manager, start_idx)
    
    # Traitement parallèle par chunks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk_tile_ids, i): i 
            for i, chunk_tile_ids in enumerate(chunks)
        }
        
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            processed_chunks += 1
            
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
                
                # Statistiques intermédiaires
                successful = len([r for r in chunk_results if "✅" in r])
                skipped = len([r for r in chunk_results if "⏭️" in r])
                failed = len([r for r in chunk_results if "❌" in r])
                
                elapsed = time.time() - start_time
                eta = (elapsed / processed_chunks) * (len(chunks) - processed_chunks)
                
                logger.info(f"📊 Chunk {chunk_id+1}/{len(chunks)} | ✅{successful} ⏭️{skipped} ❌{failed} | ETA: {eta/60:.1f}min")
                
            except Exception as e:
                logger.error(f"❌ Erreur chunk {chunk_id}: {e}")
                all_results.extend([f"❌ Chunk {chunk_id}: {str(e)}"])
    
    # Statistiques finales
    end_time = time.time()
    processing_time = end_time - start_time
    successful = len([r for r in all_results if "✅" in r])
    skipped = len([r for r in all_results if "⏭️" in r])
    failed = len([r for r in all_results if "❌" in r])
    
    logger.info(f"""
🏁 TRAITEMENT INDIVIDUEL TERMINÉ!
════════════════════════════════════════════════════════════════════════════════
⏱️  Temps total: {processing_time/3600:.1f}h ({processing_time/60:.1f}min)
🏎️  Vitesse: {len(all_tile_ids)/processing_time:.2f} tuiles/s
📊 Résultats:
   ✅ Succès: {successful:,}/{len(all_tile_ids):,} tuiles ({successful/len(all_tile_ids)*100:.1f}%)
   ⏭️  Ignorées: {skipped:,} tuiles
   ❌ Échecs: {failed:,} tuiles
════════════════════════════════════════════════════════════════════════════════
    """)
    
    # Sauvegarder résumé
    summary = {
        'total_tiles': len(all_tile_ids),
        'processing_time_hours': processing_time / 3600,
        'tiles_per_second': len(all_tile_ids) / processing_time,
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'chunk_size': chunk_size,
        'max_workers': max_workers,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = f"{s3_output_prefix.rstrip('/')}/individual_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    s3_manager.save_json(summary, summary_path)
    
    return all_results

if __name__ == "__main__":
    # 🚀 CONFIGURATION POUR TRAITEMENT INDIVIDUEL
    s3_input_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128/"
    s3_output_prefix = "s3://antoinelesauvage/vergers-france/preds_sem/"
    
    # Paramètres optimisés pour traitement individuel
    chunk_size = 70   # Tuiles par chunk (pour parallélisme I/O)
    max_workers = 3   # Workers pour les chunks
    
    # LANCEMENT
    results = run_individual_inference(
        s3_input_prefix=s3_input_prefix,
        s3_output_prefix=s3_output_prefix,
        chunk_size=chunk_size,
        max_workers=max_workers
    )
    
    print("🎉 Traitement individuel terminé!")