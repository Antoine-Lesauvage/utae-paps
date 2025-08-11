import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import s3fs
import json
from src import model_utils
from types import SimpleNamespace
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
import concurrent.futures
from tqdm import tqdm
import logging
from functools import partial
import time
import threading
import queue

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
with open("../UTAE_PAPs/conf.json") as f:
    config_dict = json.load(f)

with open("NORM_S2_patch.json") as f:
    norm_vals = json.load(f)

config = SimpleNamespace(**config_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class S3Manager:
    """Gestionnaire des connexions S3 thread-safe"""
    def __init__(self):
        os.environ["AWS_ACCESS_KEY_ID"] = 'XPWMMMXAZ529WAY25TST'
        os.environ["AWS_SECRET_ACCESS_KEY"] = 'C7cvdpblbUjBjdxZiJm9ouRLfe76xLZ2znD+8afV'
        os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJYUFdNTU1YQVo1MjlXQVkyNVRTVCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0NTU1NzU2LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1MTYxMzM1LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDU1NjUzNCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDpmMmE4NTdjOC0yNzdlLTUwZTItZjkyMS0yN2MzY2ZmNDkwNzciLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI4ZTYwMmU2Ny03OTU4LTQ0NTctYmFlMS0zZGQ4MTJiMmRkNjYiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.lXZfZ0BAlprfKzi3hKUidQJRXICtNyoVlupjIxlBuFxWI8TtE58oL9yodV5hkUcXZWctPPx9abqWnnD_SWo0Bg'
        os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
        
        self._fs = None
    
    @property
    def fs(self):
        """Lazy loading du filesystem S3"""
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(
                client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
                key=os.environ["AWS_ACCESS_KEY_ID"],
                secret=os.environ["AWS_SECRET_ACCESS_KEY"],
                token=os.environ["AWS_SESSION_TOKEN"]
            )
        return self._fs
    
    def list_tiles(self, bucket_prefix):
        """Liste tous les tiles disponibles"""
        try:
            files = self.fs.ls(bucket_prefix)
            tile_ids = []
            for file_path in files:
                if file_path.endswith('.npy'):
                    tile_id = os.path.basename(file_path).replace('.npy', '')
                    tile_ids.append(tile_id)
            return sorted(tile_ids)
        except Exception as e:
            logger.error(f"Erreur lors du listing des tiles: {e}")
            return []
    
    def list_processed_tiles(self, output_prefix):
        """Liste les tiles déjà traités"""
        try:
            files = self.fs.ls(output_prefix)
            processed_tiles = set()
            for file_path in files:
                if '_best_fold_' in file_path and file_path.endswith('.npy'):
                    # Extraire le tile_id du nom de fichier
                    filename = os.path.basename(file_path)
                    # Format: tile_id_best_fold_X.npy
                    tile_id = filename.split('_best_fold_')[0]
                    processed_tiles.add(tile_id)
            return processed_tiles
        except Exception as e:
            logger.warning(f"Erreur lors du listing des tiles traités: {e}")
            return set()
    
    def load_tile(self, s3_path):
        """Charge un tile depuis S3"""
        try:
            with self.fs.open(s3_path, 'rb') as f:
                return np.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {s3_path}: {e}")
            return None
    
    def save_result(self, data, s3_path):
        """Sauvegarde un résultat sur S3"""
        try:
            with self.fs.open(s3_path, 'wb') as f:
                np.save(f, data)
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde vers {s3_path}: {e}")
            return False

class ProgressTracker:
    """Gestionnaire de progression avec sauvegarde"""
    def __init__(self, output_prefix, s3_manager):
        self.output_prefix = output_prefix
        self.s3_manager = s3_manager
        self.progress_file = f"{output_prefix}/progress_tracking.json"
        
    def get_remaining_tiles(self, input_prefix):
        """Détermine les tiles restant à traiter"""
        logger.info("🔍 Analyse des tiles déjà traités...")
        
        # Lister tous les tiles d'entrée
        all_tiles = set(self.s3_manager.list_tiles(input_prefix))
        logger.info(f"📊 {len(all_tiles)} tiles d'entrée trouvés")
        
        # Lister les tiles déjà traités
        processed_tiles = self.s3_manager.list_processed_tiles(self.output_prefix)
        logger.info(f"✅ {len(processed_tiles)} tiles déjà traités")
        
        # Calculer les tiles restants
        remaining_tiles = all_tiles - processed_tiles
        logger.info(f"🔄 {len(remaining_tiles)} tiles restant à traiter")
        
        if len(remaining_tiles) == 0:
            logger.info("🎉 Tous les tiles sont déjà traités !")
            return []
        
        # Afficher quelques exemples
        if len(processed_tiles) > 0:
            logger.info(f"Exemples traités: {list(list(processed_tiles)[:5])}")
        if len(remaining_tiles) > 0:
            logger.info(f"Exemples restants: {list(list(remaining_tiles)[:5])}")
        
        return sorted(list(remaining_tiles))
    
    def save_progress(self, stats):
        """Sauvegarde les statistiques de progression"""
        try:
            progress_data = {
                'timestamp': time.time(),
                'stats': stats
            }
            with self.s3_manager.fs.open(self.progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder la progression: {e}")

# Reprise du code précédent pour les classes GPUModelManager, etc.
class GPUModelManager:
    """Gestionnaire des modèles optimisé pour GPU - FP32 uniquement pour la précision"""
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def load_models(self):
        """Charge tous les modèles en mémoire GPU en FP32"""
        if self.loaded:
            return
        
        logger.info(f"🔄 Chargement des modèles sur GPU {self.gpu_id} (FP32 pour précision)")
        
        with torch.cuda.device(self.gpu_id):
            for fold_num in range(1, 6):
                try:
                    model = model_utils.get_model(config, mode="panoptic")
                    model.to(self.device)
                    
                    checkpoint_path = os.path.join("../UTAE_PAPs", f"Fold_{fold_num}", "model.pth.tar")
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        model.load_state_dict(checkpoint["state_dict"])
                        model.eval()
                        model = model.float()
                        self.models[fold_num] = model
                        logger.info(f"✅ Modèle Fold_{fold_num} chargé en FP32")
                    else:
                        logger.warning(f"⚠️ Modèle Fold_{fold_num} introuvable")
                except Exception as e:
                    logger.error(f"❌ Erreur chargement Fold_{fold_num}: {e}")
        
        self.loaded = True
        logger.info(f"📦 {len(self.models)} modèles chargés sur GPU {self.gpu_id}")
    
    def get_model(self, fold_num):
        if not self.loaded:
            self.load_models()
        return self.models.get(fold_num)

def calculate_confidence_optimized(output):
    """Version optimisée du calcul de confiance"""
    try:
        semantic_key = None
        for key in ["pano_semantic", "semantic", "heatmap"]:
            if key in output and output[key] is not None:
                semantic_key = key
                break
        
        if semantic_key is None:
            return 0.0
        
        semantic_logits = output[semantic_key]
        if not torch.is_tensor(semantic_logits):
            return 0.0
        
        with torch.no_grad():
            if len(semantic_logits.shape) == 4:  # (B, C, H, W)
                if semantic_logits.shape[1] > 1:  # Multi-classes
                    probs = F.softmax(semantic_logits, dim=1)
                    max_probs = torch.max(probs, dim=1)[0]  # (B, H, W)
                    confidence = max_probs.mean().item()
                else:
                    confidence = 0.8
            else:
                confidence = 0.8
            
        return confidence
    except Exception as e:
        logger.debug(f"Erreur calcul confiance: {e}")
        return 0.0

def process_tile_inference_all_folds(tile_id, input_prefix, output_prefix, model_manager, s3_manager):
    """Traite une tuile avec tous les folds et sélectionne le meilleur"""
    try:
        start_time = time.time()
        
        # Vérifier si déjà traité (double sécurité)
        processed_tiles = s3_manager.list_processed_tiles(output_prefix)
        if tile_id in processed_tiles:
            return f"⏭️  {tile_id} déjà traité, ignoré"
        
        # Charger les données
        input_path = f"{input_prefix}/{tile_id}.npy"
        arr = s3_manager.load_tile(input_path)
        if arr is None:
            return f"❌ Échec chargement {tile_id}"
        
        x = torch.from_numpy(arr).float().unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda(model_manager.device)
        
        T, C, H, W = x.shape[1:]
        batch_positions = torch.linspace(0, 1, steps=T).unsqueeze(0).to(x.device)
        
        # Stocker les résultats de tous les folds
        fold_results = {}
        fold_confidences = {}
        
        # Inférence avec chaque fold
        with torch.cuda.device(model_manager.gpu_id):
            for fold in range(1, 6):
                model = model_manager.get_model(fold)
                if model is None:
                    continue
                
                try:
                    mean = torch.tensor(norm_vals[f"Fold_{fold}"]["mean"]).float().to(x.device)
                    std = torch.tensor(norm_vals[f"Fold_{fold}"]["std"]).float().to(x.device)
                    mean = mean.reshape(1, -1, 1, 1)
                    std = std.reshape(1, -1, 1, 1)
                    x_norm = (x - mean) / std
                    
                    with torch.no_grad():
                        output = model(x_norm, batch_positions=batch_positions)
                        confidence = calculate_confidence_optimized(output)
                        
                        fold_results[fold] = {
                            k: v.cpu().numpy() if torch.is_tensor(v) else v 
                            for k, v in output.items()
                        }
                        fold_confidences[fold] = confidence
                        
                except Exception as e:
                    logger.warning(f"Erreur fold {fold} pour {tile_id}: {e}")
                    continue
        
        if not fold_confidences:
            return f"❌ Aucune prédiction valide pour {tile_id}"
        
        # Sélectionner le meilleur fold
        best_fold = max(fold_confidences, key=fold_confidences.get)
        best_confidence = fold_confidences[best_fold]
        best_result = fold_results[best_fold]
        
        best_result['best_fold'] = best_fold
        best_result['confidence'] = best_confidence
        best_result['all_confidences'] = fold_confidences
        best_result['tile_id'] = tile_id
        
        # Sauvegarder le résultat
        output_path = f"{output_prefix}/{tile_id}_best_fold_{best_fold}.npy"
        success = s3_manager.save_result(best_result, output_path)
        
        processing_time = time.time() - start_time
        
        if success:
            return f"✅ {tile_id} → Fold_{best_fold} (conf: {best_confidence:.3f}) [{processing_time:.1f}s]"
        else:
            return f"❌ Échec sauvegarde {tile_id}"
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {tile_id}: {e}")
        return f"❌ Erreur {tile_id}: {str(e)}"

class GPUWorker:
    """Worker qui traite les tuiles sur un GPU spécifique"""
    def __init__(self, gpu_id, input_prefix, output_prefix):
        self.gpu_id = gpu_id
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.model_manager = None
        self.s3_manager = None
        
    def initialize(self):
        """Initialise le worker"""
        self.model_manager = GPUModelManager(self.gpu_id)
        self.s3_manager = S3Manager()
        logger.info(f"🚀 Worker GPU {self.gpu_id} initialisé")
    
    def process_tiles(self, tile_ids):
        """Traite une liste de tuiles"""
        if self.model_manager is None:
            self.initialize()
        
        results = []
        for tile_id in tile_ids:
            result = process_tile_inference_all_folds(
                tile_id, self.input_prefix, self.output_prefix,
                self.model_manager, self.s3_manager
            )
            results.append(result)
            logger.info(result)
        
        return results

def create_batches(items, batch_size):
    """Crée des batches d'éléments"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def run_parallel_inference_with_resume():
    """Lance l'inférence avec reprise automatique"""
    input_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128"
    output_prefix = "s3://antoinelesauvage/vergers-france/preds"
    
    # Initialiser les gestionnaires
    s3_manager = S3Manager()
    progress_tracker = ProgressTracker(output_prefix, s3_manager)
    
    # Déterminer les tiles restant à traiter
    remaining_tiles = progress_tracker.get_remaining_tiles(input_prefix)
    
    if not remaining_tiles:
        logger.info("🎉 Tous les tiles sont déjà traités !")
        return
    
    logger.info(f"🚀 Reprise du traitement pour {len(remaining_tiles)} tiles restants")
    
    # Configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tiles_per_batch = 15
    
    logger.info(f"🖥️  {num_gpus} GPU(s) détecté(s)")
    logger.info(f"📦 {tiles_per_batch} tuiles par batch")
    
    # Traitement avec un seul GPU
    if num_gpus == 1:
        worker = GPUWorker(0, input_prefix, output_prefix)
        worker.initialize()
        
        tile_batches = list(create_batches(remaining_tiles, tiles_per_batch))
        start_time = time.time()
        
        all_results = []
        with tqdm(total=len(remaining_tiles), desc="Tuiles restantes") as pbar:
            for i, batch in enumerate(tile_batches):
                batch_results = worker.process_tiles(batch)
                all_results.extend(batch_results)
                pbar.update(len(batch))
                
                # Statistiques temps réel
                elapsed = time.time() - start_time
                speed = len(all_results) / elapsed if elapsed > 0 else 0
                successful = len([r for r in all_results if "✅" in r])
                skipped = len([r for r in all_results if "⏭️" in r])
                
                pbar.set_postfix({
                    'Batch': f"{i+1}/{len(tile_batches)}",
                    'Vitesse': f"{speed:.1f} t/s",
                    'Nouveaux': f"{successful}",
                    'Ignorés': f"{skipped}"
                })
                
                # Sauvegarder la progression périodiquement
                if (i + 1) % 10 == 0:
                    stats = {
                        'processed_batches': i + 1,
                        'total_batches': len(tile_batches),
                        'successful': successful,
                        'skipped': skipped,
                        'failed': len([r for r in all_results if "❌" in r]),
                        'speed': speed
                    }
                    progress_tracker.save_progress(stats)
    
    # Statistiques finales
    end_time = time.time()
    processing_time = end_time - start_time
    successful = len([r for r in all_results if "✅" in r])
    skipped = len([r for r in all_results if "⏭️" in r])
    failed = len([r for r in all_results if "❌" in r])
    
    logger.info(f"🏁 Traitement de reprise terminé en {processing_time:.2f} secondes")
    logger.info(f"✅ {successful} nouveaux tiles traités")
    logger.info(f"⏭️  {skipped} tiles déjà traités (ignorés)")
    logger.info(f"❌ {failed} échecs")
    
    if successful > 0:
        logger.info(f"⚡ Vitesse moyenne: {successful/processing_time:.2f} nouveaux tiles/seconde")
    
    # Rapport final
    final_stats = {
        'resume_session': True,
        'remaining_tiles_at_start': len(remaining_tiles),
        'newly_processed': successful,
        'already_processed_skipped': skipped,
        'failed': failed,
        'processing_time': processing_time,
        'timestamp': time.time()
    }
    progress_tracker.save_progress(final_stats)

def check_progress_status():
    """Fonction utilitaire pour vérifier l'état actuel"""
    input_prefix = "s3://antoinelesauvage/vergers-france/patches_2018_128"
    output_prefix = "s3://antoinelesauvage/vergers-france/preds"
    
    s3_manager = S3Manager()
    progress_tracker = ProgressTracker(output_prefix, s3_manager)
    
    # Analyser la progression
    all_tiles = set(s3_manager.list_tiles(input_prefix))
    processed_tiles = s3_manager.list_processed_tiles(output_prefix)
    remaining_tiles = all_tiles - processed_tiles
    
    print(f"📊 ÉTAT ACTUEL:")
    print(f"   Total tiles: {len(all_tiles)}")
    print(f"   Traités: {len(processed_tiles)}")
    print(f"   Restants: {len(remaining_tiles)}")
    print(f"   Progression: {(len(processed_tiles)/len(all_tiles)*100):.1f}%")
    
    if len(remaining_tiles) > 0:
        print(f"   Premiers restants: {list(list(remaining_tiles)[:10])}")
    
    return len(remaining_tiles)

if __name__ == "__main__":
    # Vérifier l'état actuel
    remaining_count = check_progress_status()
    
    if remaining_count > 0:
        print(f"\n🚀 Lancement du traitement pour {remaining_count} tiles restants...")
        run_parallel_inference_with_resume()
    else:
        print("\n🎉 Tous les tiles sont déjà traités !")