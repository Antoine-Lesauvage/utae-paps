import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import s3fs
import os

# Configuration S3 (identique)
os.environ["AWS_ACCESS_KEY_ID"] = 'ZDEMRP59PONSWI7XOOHA'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'vl4S7UvZp2N+MZ8AXVy6+nCb20QjvzQHrBBlgdFJ'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJaREVNUlA1OVBPTlNXSTdYT09IQSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0NDYyODUxLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1MDcwODA0LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDQ2NjAwNCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDplNzkwYjU1YS05NTZkLTlmOTgtMWYzNy03MTdiYzAzMjdjNzQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiI2ZWQ5MzUwZi1hZjA0LTQxNzYtYTRkZi1lNzllZDNmM2YzMmMiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.zmYq-lXGCd628NwEnKcXMsUiXjq3C8cqJA-etTyK5x3fHCTwTSZNmksqKgmcg59b7l_1O1362s8Gv1dXEEUNCw'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'

# Colormap
try:
    cm = matplotlib.colormaps.get_cmap('tab20')
except AttributeError:
    cm = matplotlib.cm.get_cmap('tab20')

def_colors = cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1,19)]+['w']
cmap = ListedColormap(colors = cus_colors, name='agri',N=20)

def load_multi_fold_inference(file_path):
    """Charge les résultats d'inférence multi-fold"""
    data = np.load(file_path, allow_pickle=True).item()
    
    print("🔍 Analyse du fichier multi-fold:")
    print(f"   Clés principales: {list(data.keys())}")
    
    if 'best_fold' in data:
        print(f"   Meilleur fold: {data['best_fold']}")
    
    if 'fold_results' in data:
        available_folds = list(data['fold_results'].keys())
        print(f"   Folds disponibles: {available_folds}")
        
        # Analyser un fold exemple
        if available_folds:
            example_fold = available_folds[0]
            fold_data = data['fold_results'][example_fold]
            print(f"   Structure fold {example_fold}:")
            for key, value in fold_data.items():
                if hasattr(value, 'shape'):
                    print(f"     {key}: shape = {value.shape}, dtype = {value.dtype}")
                elif isinstance(value, list):
                    print(f"     {key}: liste de {len(value)} éléments")
                else:
                    print(f"     {key}: {type(value)}")
    
    if 'fold_confidences' in data:
        fold_confidences = data['fold_confidences']
        print(f"   Confidences par fold: {fold_confidences}")
    
    return data

def load_pastis_annotations_complete(tile_id):
    """Charge toutes les annotations PASTIS (identique à avant)"""
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"], 
        secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
        token=os.environ["AWS_SESSION_TOKEN"]
    )
    
    pastis_root = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS"
    
    annotations = {}
    annotation_files = {
        'PARCELID': f"{pastis_root}/ANNOTATIONS/ParcelIDs_{tile_id}.npy",
        'TARGET': f"{pastis_root}/ANNOTATIONS/TARGET_{tile_id}.npy", 
        'HEATMAP': f"{pastis_root}/INSTANCE_ANNOTATIONS/HEATMAP_{tile_id}.npy",
        'INSTANCES': f"{pastis_root}/INSTANCE_ANNOTATIONS/INSTANCES_{tile_id}.npy",
        'ZONES': f"{pastis_root}/INSTANCE_ANNOTATIONS/ZONES_{tile_id}.npy"
    }
    
    print(f"📁 Chargement annotations complètes tuile {tile_id}...")
    
    for name, path in annotation_files.items():
        try:
            with fs.open(path, 'rb') as f:
                data = np.load(f)
                annotations[name] = data
                print(f"   ✅ {name}: {data.shape}, dtype: {data.dtype}")
        except Exception as e:
            print(f"   ❌ Erreur {name}: {e}")
            annotations[name] = None
    
    return annotations

def create_prediction_masks_for_fold(fold_data, fold_num, confidence_threshold=0.3):
    """Crée les masques de prédiction pour un fold spécifique"""
    
    results = {'fold_num': fold_num}
    
    # 1. Masque sémantique panoptique
    if 'pano_semantic' in fold_data:
        pano_sem = fold_data['pano_semantic']
        if len(pano_sem.shape) == 4:  # (1, 20, 128, 128)
            results['semantic_mask'] = np.argmax(pano_sem[0], axis=0)
        elif len(pano_sem.shape) == 3:  # (20, 128, 128)
            results['semantic_mask'] = np.argmax(pano_sem, axis=0)
        else:
            results['semantic_mask'] = pano_sem
        print(f"   📊 Fold {fold_num} - Masque sémantique: {results['semantic_mask'].shape}")
    
    # 2. Masque d'instances panoptique
    if 'pano_instance' in fold_data:
        pano_inst = fold_data['pano_instance']
        if len(pano_inst.shape) == 3:  # (1, 128, 128)
            results['instance_mask'] = pano_inst[0]
        else:
            results['instance_mask'] = pano_inst
        print(f"   📊 Fold {fold_num} - Masque instances: {results['instance_mask'].shape}")
    
    # 3. Instances individuelles
    results['individual_instances'] = []
    
    if 'instance_masks' in fold_data and 'confidence' in fold_data:
        instance_masks = fold_data['instance_masks']
        confidences = fold_data['confidence']
        
        if len(confidences.shape) > 1:
            confidences = confidences.flatten()
        
        # Filtrage par confiance
        for i, (mask, conf) in enumerate(zip(instance_masks, confidences)):
            if conf >= confidence_threshold:
                instance_info = {
                    'id': i,
                    'mask': mask,
                    'confidence': conf
                }
                
                # Boîtes englobantes si disponibles
                if 'instance_boxes' in fold_data and i < len(fold_data['instance_boxes']):
                    instance_info['bbox'] = fold_data['instance_boxes'][i]
                
                results['individual_instances'].append(instance_info)
        
        print(f"   📊 Fold {fold_num} - Instances filtrées: {len(results['individual_instances'])}/{len(instance_masks)}")
    
    # 4. Heatmaps et autres
    for key in ['heatmap', 'saliency', 'center_mask']:
        if key in fold_data:
            data = fold_data[key]
            if len(data.shape) == 4:  # (1, 1, 128, 128)
                results[key] = data[0, 0]
            elif len(data.shape) == 3:  # (1, 128, 128)
                results[key] = data[0]
            else:
                results[key] = data
    
    return results

def visualize_multi_fold_comparison(tile_id, multi_fold_file):
    """Visualisation comparative de tous les folds"""
    
    print(f"🎨 VISUALISATION MULTI-FOLD - Tuile {tile_id}")
    print("="*80)
    
    # 1. Charger les données
    multi_fold_data = load_multi_fold_inference(multi_fold_file)
    annotations = load_pastis_annotations_complete(tile_id)
    
    # 2. Traiter chaque fold
    fold_results = {}
    available_folds = list(multi_fold_data['fold_results'].keys())
    
    print(f"\n🔄 Traitement des folds: {available_folds}")
    for fold_num in available_folds:
        fold_data = multi_fold_data['fold_results'][fold_num]
        fold_results[fold_num] = create_prediction_masks_for_fold(fold_data, fold_num)
    
    # 3. Créer la visualisation
    n_folds = len(available_folds)
    best_fold = multi_fold_data.get('best_fold', available_folds[0])
    
    # Figure principale: comparaison sémantique de tous les folds
    fig1, axes1 = plt.subplots(2, n_folds + 1, figsize=(4*(n_folds+1), 8))
    fig1.suptitle(f'Comparaison Sémantique Multi-Fold - Tuile {tile_id}', fontsize=16, fontweight='bold')
    
    if n_folds == 1:
        axes1 = axes1.reshape(2, -1)
    
    # Ligne 1: Ground Truth + Prédictions de chaque fold
    # GT TARGET
    col = 0
    if annotations.get('TARGET') is not None:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]
        
        im = axes1[0, col].imshow(target_data, cmap=cmap, vmin=0, vmax=19)
        axes1[0, col].set_title(f'Ground Truth\nTARGET')
        axes1[0, col].axis('off')
        plt.colorbar(im, ax=axes1[0, col], fraction=0.046)
    else:
        axes1[0, col].text(0.5, 0.5, 'GT\nIndisponible', ha='center', va='center')
        axes1[0, col].set_title('Ground Truth')
        axes1[0, col].axis('off')
    
    # Prédictions pour chaque fold
    for i, fold_num in enumerate(available_folds):
        col = i + 1
        fold_result = fold_results[fold_num]
        
        if 'semantic_mask' in fold_result:
            im = axes1[0, col].imshow(fold_result['semantic_mask'], cmap=cmap, vmin=0, vmax=19)
            n_classes = len(np.unique(fold_result['semantic_mask']))
            fold_conf = multi_fold_data.get('fold_confidences', {}).get(fold_num, 'N/A')
            
            title = f'Fold {fold_num}\n{n_classes} classes'
            if fold_num == best_fold:
                title += '\n⭐ MEILLEUR'
            if fold_conf != 'N/A':
                title += f'\nConf: {fold_conf:.3f}'
            
            axes1[0, col].set_title(title, fontweight='bold' if fold_num == best_fold else 'normal')
            axes1[0, col].axis('off')
            plt.colorbar(im, ax=axes1[0, col], fraction=0.046)
        else:
            axes1[0, col].text(0.5, 0.5, f'Fold {fold_num}\nErreur', ha='center', va='center')
            axes1[0, col].set_title(f'Fold {fold_num}')
            axes1[0, col].axis('off')
    
    # Ligne 2: Instances + détections
    # GT INSTANCES
    col = 0
    if annotations.get('INSTANCES') is not None:
        im = axes1[1, col].imshow(annotations['INSTANCES'], cmap='nipy_spectral')
        n_instances = len(np.unique(annotations['INSTANCES'])) - 1
        axes1[1, col].set_title(f'GT Instances\n{n_instances} parcelles')
        axes1[1, col].axis('off')
        plt.colorbar(im, ax=axes1[1, col], fraction=0.046)
    else:
        axes1[1, col].text(0.5, 0.5, 'GT Instances\nIndisponible', ha='center', va='center')
        axes1[1, col].set_title('GT Instances')
        axes1[1, col].axis('off')
    
    # Instances prédites pour chaque fold
    for i, fold_num in enumerate(available_folds):
        col = i + 1
        fold_result = fold_results[fold_num]
        
        if 'instance_mask' in fold_result:
            axes1[1, col].imshow(fold_result['instance_mask'], cmap='nipy_spectral')
            
            # Ajouter les boîtes des détections
            n_detections = len(fold_result['individual_instances'])
            for inst in fold_result['individual_instances']:
                if 'bbox' in inst:
                    x1, y1, x2, y2 = inst['bbox']
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, 
                                           linewidth=1, edgecolor='red', 
                                           facecolor='none', alpha=0.7)
                    axes1[1, col].add_patch(rect)
            
            title = f'Fold {fold_num}\n{n_detections} détections'
            if fold_num == best_fold:
                title += '\n⭐'
            
            axes1[1, col].set_title(title, fontweight='bold' if fold_num == best_fold else 'normal')
        else:
            axes1[1, col].text(0.5, 0.5, f'Fold {fold_num}\nPas d\'instances', ha='center', va='center')
            axes1[1, col].set_title(f'Fold {fold_num}')
        
        axes1[1, col].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder la comparaison principale
    output_file1 = f'multi_fold_comparison_tile_{tile_id}.png'
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"💾 Comparaison multi-fold sauvegardée: {output_file1}")
    plt.show()
    
    # 4. Figure détaillée pour le meilleur fold
    print(f"\n🏆 Analyse détaillée du meilleur fold: {best_fold}")
    
    best_fold_result = fold_results[best_fold]
    visualize_detailed_fold_analysis(tile_id, best_fold, best_fold_result, annotations)
    
    # 5. Statistiques comparatives
    print_multi_fold_statistics(tile_id, fold_results, multi_fold_data, annotations)
    
    return fold_results, multi_fold_data

def visualize_detailed_fold_analysis(tile_id, fold_num, fold_result, annotations):
    """Analyse détaillée d'un fold spécifique (similaire à la visualisation simple)"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Analyse Détaillée - Tuile {tile_id}, Fold {fold_num}', fontsize=16, fontweight='bold')
    
    # Ligne 1: Comparaison sémantique
    # GT
    if annotations.get('TARGET') is not None:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]
        axes[0, 0].imshow(target_data, cmap=cmap, vmin=0, vmax=19)
        axes[0, 0].set_title('Ground Truth\nSémantique')
    else:
        axes[0, 0].text(0.5, 0.5, 'GT\nIndisponible', ha='center', va='center')
        axes[0, 0].set_title('GT Sémantique')
    axes[0, 0].axis('off')
    
    # Prédiction
    if 'semantic_mask' in fold_result:
        axes[0, 1].imshow(fold_result['semantic_mask'], cmap=cmap, vmin=0, vmax=19)
        axes[0, 1].set_title(f'Prédiction Fold {fold_num}\nSémantique')
    else:
        axes[0, 1].text(0.5, 0.5, 'Prédiction\nIndisponible', ha='center', va='center')
        axes[0, 1].set_title('Prédiction Sémantique')
    axes[0, 1].axis('off')
    
    # Différence
    if annotations.get('TARGET') is not None and 'semantic_mask' in fold_result:
        diff_mask = (target_data != fold_result['semantic_mask']).astype(float)
        accuracy = 1 - np.mean(diff_mask)
        axes[0, 2].imshow(diff_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Différences\nAccuracy: {accuracy:.3f}')
    else:
        axes[0, 2].text(0.5, 0.5, 'Comparaison\nImpossible', ha='center', va='center')
        axes[0, 2].set_title('Différences')
    axes[0, 2].axis('off')
    
    # Ligne 2: Instances et heatmaps
    # Instances GT
    if annotations.get('INSTANCES') is not None:
        axes[1, 0].imshow(annotations['INSTANCES'], cmap='nipy_spectral')
        axes[1, 0].set_title('GT Instances')
    else:
        axes[1, 0].text(0.5, 0.5, 'GT Instances\nIndisponible', ha='center', va='center')
        axes[1, 0].set_title('GT Instances')
    axes[1, 0].axis('off')
    
    # Instances prédites
    if 'instance_mask' in fold_result:
        axes[1, 1].imshow(fold_result['instance_mask'], cmap='nipy_spectral')
        axes[1, 1].set_title(f'Instances Fold {fold_num}')
    else:
        axes[1, 1].text(0.5, 0.5, 'Instances\nIndisponible', ha='center', va='center')
        axes[1, 1].set_title('Instances Prédites')
    axes[1, 1].axis('off')
    
    # Heatmap
    if 'heatmap' in fold_result:
        im = axes[1, 2].imshow(fold_result['heatmap'], cmap='hot')
        axes[1, 2].set_title(f'Heatmap Fold {fold_num}')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 2].text(0.5, 0.5, 'Heatmap\nIndisponible', ha='center', va='center')
        axes[1, 2].set_title('Heatmap')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_file = f'detailed_fold_{fold_num}_tile_{tile_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Analyse détaillée fold {fold_num} sauvegardée: {output_file}")
    plt.show()

def print_multi_fold_statistics(tile_id, fold_results, multi_fold_data, annotations):
    """Affiche des statistiques comparatives entre les folds"""
    
    print(f"\n📊 STATISTIQUES COMPARATIVES - Tuile {tile_id}")
    print("="*70)
    
    # En-tête du tableau
    print("Fold | Détections | Classes Pred | Conf Moy | Conf Max | Score Global")
    print("-"*70)
    
    fold_scores = {}
    
    for fold_num, fold_result in fold_results.items():
        n_detections = len(fold_result.get('individual_instances', []))
        
        # Confidences
        confidences = [inst['confidence'] for inst in fold_result.get('individual_instances', [])]
        conf_mean = np.mean(confidences) if confidences else 0
        conf_max = np.max(confidences) if confidences else 0
        
        # Classes prédites
        n_pred_classes = len(np.unique(fold_result.get('semantic_mask', [0])))
        
        # Score global (combinaison de métriques)
        fold_conf_global = multi_fold_data.get('fold_confidences', {}).get(fold_num, 0)
        score = (conf_mean + fold_conf_global + (n_detections/100)) / 3  # Score arbitraire
        
        fold_scores[fold_num] = score
        
        # Marquer le meilleur fold
        marker = "⭐" if fold_num == multi_fold_data.get('best_fold') else "  "
        
        print(f" {fold_num}   |    {n_detections:4d}    |      {n_pred_classes:2d}      | {conf_mean:8.3f} | {conf_max:8.3f} |   {score:8.3f} {marker}")
    
    # Ground Truth pour comparaison
    if annotations.get('INSTANCES') is not None:
        n_gt_instances = len(np.unique(annotations['INSTANCES'])) - 1
        print(f"\n🎯 Ground Truth: {n_gt_instances} instances")
    
    if annotations.get('TARGET') is not None:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]
        n_gt_classes = len(np.unique(target_data))
        print(f"🎯 Ground Truth: {n_gt_classes} classes sémantiques")
    
    # Recommandation
    best_by_score = max(fold_scores, key=fold_scores.get) if fold_scores else None
    official_best = multi_fold_data.get('best_fold')
    
    print(f"\n🏆 RECOMMANDATIONS:")
    print(f"   Meilleur selon le modèle: Fold {official_best}")
    if best_by_score and best_by_score != official_best:
        print(f"   Meilleur selon les détections: Fold {best_by_score}")
    print(f"   Consensus: Fold {official_best} {'✅' if official_best == best_by_score else '⚠️'}")

# UTILISATION PRINCIPALE
def main():
    tile_id = "10000"
    multi_fold_file = "preds/S2_10000_all_folds_analysis.npy"  # Ajustez le nom de votre fichier
    
    fold_results, multi_fold_data = visualize_multi_fold_comparison(tile_id, multi_fold_file)
    
    return fold_results, multi_fold_data

if __name__ == "__main__":
    results = main()