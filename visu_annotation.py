import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import s3fs
import os

# Configuration S3 (vos identifiants)
os.environ["AWS_ACCESS_KEY_ID"] = '1CH53NXGI0M8EML05AQY'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'YCz+C1hM3kokhoJTmISBFV84VHqpgmw4dUZ6Y5pk'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiIxQ0g1M05YR0kwTThFTUwwNUFRWSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU0Mzc5NDA3LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU0OTg0MzUzLCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NDM3OTU1MiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDpmNTBmMDZkNC1kZWU1LWE4N2QtY2FhNC01MTg5NTkyZDVjZWQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJiODNkZWZiNC1kOWQwLTRhNWYtYjM3Ny1jMjI5NGZiYmMwZGEiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.v9xea_8hrBrC-X5PDukdL3hvn96XkN1cbmu9fHIlHd9_Hhe3OD68tjcCpfXdK22AvoKrrRCFseRjfyCwKzm--w'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'

# Correction du warning matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Colormap - correction du warning
try:
    cm = matplotlib.colormaps.get_cmap('tab20')  # Nouvelle syntaxe
except AttributeError:
    cm = matplotlib.cm.get_cmap('tab20')  # Ancienne syntaxe pour compatibilité

def_colors = cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1,19)]+['w']
cmap = ListedColormap(colors = cus_colors, name='agri',N=20)

def load_inference_results(file_path):
    """Charge les résultats d'inférence"""
    data = np.load(file_path, allow_pickle=True).item()
    
    print("🔍 Analyse du fichier d'inférence:")
    print(f"   Clés disponibles: {list(data.keys())}")
    
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: shape = {value.shape}, dtype = {value.dtype}")
        elif isinstance(value, list):
            print(f"   {key}: liste de {len(value)} éléments")
        else:
            print(f"   {key}: {type(value)}")
    
    return data

def load_pastis_annotations_complete(tile_id):
    """Charge toutes les annotations PASTIS disponibles"""
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"], 
        secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
        token=os.environ["AWS_SESSION_TOKEN"]
    )
    
    pastis_root = "s3://antoinelesauvage/zenodo/PASTIS_unz/PASTIS"
    
    annotations = {}
    
    # Liste des fichiers d'annotation à charger
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
                
                # Infos supplémentaires selon le type
                if name == 'PARCELID':
                    unique_parcels = np.unique(data)
                    print(f"      Parcelles uniques: {len(unique_parcels)} (IDs: {unique_parcels[:10]}...)")
                elif name == 'TARGET':
                    unique_targets = np.unique(data)
                    print(f"      Classes cibles: {unique_targets}")
                elif name == 'INSTANCES':
                    unique_instances = np.unique(data)
                    print(f"      Instances uniques: {len(unique_instances)-1} (excl. background)")
                elif name in ['HEATMAP', 'ZONES']:
                    print(f"      Min: {data.min():.3f}, Max: {data.max():.3f}")
                    
        except Exception as e:
            print(f"   ❌ Erreur {name}: {e}")
            annotations[name] = None
    
    return annotations
def create_prediction_masks_enhanced(inference_data, confidence_threshold=0.3):
    """Crée les masques de prédiction avec analyse détaillée"""
    
    results = {}
    
    # 1. Masque sémantique panoptique
    if 'pano_semantic' in inference_data:
        pano_sem = inference_data['pano_semantic']  # (1, 20, 128, 128)
        # Prendre argmax sur la dimension des classes et supprimer la dimension batch
        results['semantic_mask'] = np.argmax(pano_sem[0], axis=0)  # (128, 128)
        print(f"📊 Masque sémantique panoptique: {pano_sem.shape} -> {results['semantic_mask'].shape}")
    elif 'semantic' in inference_data:
        semantic_logits = inference_data['semantic']
        results['semantic_mask'] = np.argmax(semantic_logits, axis=0)
        print(f"📊 Masque sémantique (logits): {results['semantic_mask'].shape}")
    
    # 2. Masque d'instances panoptique
    if 'pano_instance' in inference_data:
        pano_inst = inference_data['pano_instance']  # (1, 128, 128)
        # Supprimer la dimension batch
        results['instance_mask'] = pano_inst[0]  # (128, 128)
        print(f"📊 Masque instances panoptique: {pano_inst.shape} -> {results['instance_mask'].shape}")
    
    # 3. Instances individuelles avec métadonnées
    results['individual_instances'] = []
    
    if 'instance_masks' in inference_data and 'confidence' in inference_data:
        instance_masks = inference_data['instance_masks']
        confidences = inference_data['confidence']
        
        # Récupérer les boîtes et classes si disponibles
        boxes = inference_data.get('instance_boxes', None)
        semantic_preds = inference_data.get('semantic', None)
        
        if len(confidences.shape) > 1:
            confidences = confidences.flatten()
        
        print(f"📊 Analyse des instances individuelles:")
        print(f"   Nombre total: {len(instance_masks)}")
        print(f"   Confidences: min={confidences.min():.3f}, max={confidences.max():.3f}, moy={confidences.mean():.3f}")
        
        for i, (mask, conf) in enumerate(zip(instance_masks, confidences)):
            if conf >= confidence_threshold:
                instance_info = {
                    'id': i,
                    'mask': mask,
                    'confidence': conf
                }
                
                # Ajouter la boîte englobante si disponible
                if boxes is not None and i < len(boxes):
                    instance_info['bbox'] = boxes[i]
                
                # Prédire la classe sémantique majoritaire si logits disponibles
                if semantic_preds is not None and i < semantic_preds.shape[0]:
                    instance_info['predicted_class'] = np.argmax(semantic_preds[i])
                    instance_info['class_confidence'] = np.max(semantic_preds[i])
                
                results['individual_instances'].append(instance_info)
        
        print(f"   Instances filtrées (conf >= {confidence_threshold}): {len(results['individual_instances'])}")
    
    # 4. Heatmaps si disponibles - correction des dimensions
    for key in ['heatmap', 'saliency', 'center_mask', 'centerness']:
        if key in inference_data:
            data = inference_data[key]
            
            # Traitement spécifique selon la forme
            if key in ['heatmap', 'saliency'] and len(data.shape) == 4:  # (1, 1, 128, 128)
                results[key] = data[0, 0]  # Garder seulement (128, 128)
            elif key == 'center_mask' and len(data.shape) == 3:  # (1, 128, 128)
                results[key] = data[0]  # Garder seulement (128, 128)
            else:
                results[key] = data
            
            print(f"📊 {key.capitalize()}: {data.shape} -> {results[key].shape if hasattr(results[key], 'shape') else type(results[key])}")
    
    return results

def visualize_complete_comparison(tile_id, inference_file):
    """Visualisation complète avec toutes les annotations disponibles"""
    
    print(f"🎨 VISUALISATION COMPLÈTE - Tuile {tile_id}")
    print("="*80)
    
    # 1. Charger les données
    inference_data = load_inference_results(inference_file)
    annotations = load_pastis_annotations_complete(tile_id)
    pred_results = create_prediction_masks_enhanced(inference_data)
    
    # 2. Créer une grande figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Analyse Complète Tuile {tile_id} - Fold 5', fontsize=18, fontweight='bold')
    
    # LIGNE 1: Annotations Ground Truth
    row = 0
    
    # PARCELID
    if annotations.get('PARCELID') is not None:
        im = axes[row, 0].imshow(annotations['PARCELID'], cmap='tab20')
        n_parcels = len(np.unique(annotations['PARCELID'])) - 1
        axes[row, 0].set_title(f'PARCELID GT\n{n_parcels} parcelles')
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)
    else:
        axes[row, 0].text(0.5, 0.5, 'PARCELID\nIndisponible', ha='center', va='center')
        axes[row, 0].set_title('PARCELID GT')
    axes[row, 0].axis('off')
    
    # TARGET
    if annotations.get('TARGET') is not None:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]  # Prendre premier canal si 3D
        im = axes[row, 1].imshow(target_data, cmap=cmap, vmin=0, vmax=19)
        unique_classes = np.unique(target_data)
        axes[row, 1].set_title(f'TARGET GT\nClasses: {unique_classes}')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
    else:
        axes[row, 1].text(0.5, 0.5, 'TARGET\nIndisponible', ha='center', va='center')
        axes[row, 1].set_title('TARGET GT')
    axes[row, 1].axis('off')
    
    # INSTANCES
    if annotations.get('INSTANCES') is not None:
        im = axes[row, 2].imshow(annotations['INSTANCES'], cmap='nipy_spectral')
        n_instances = len(np.unique(annotations['INSTANCES'])) - 1
        axes[row, 2].set_title(f'INSTANCES GT\n{n_instances} instances')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
    else:
        axes[row, 2].text(0.5, 0.5, 'INSTANCES\nIndisponible', ha='center', va='center')
        axes[row, 2].set_title('INSTANCES GT')
    axes[row, 2].axis('off')
    
    # HEATMAP
    if annotations.get('HEATMAP') is not None:
        im = axes[row, 3].imshow(annotations['HEATMAP'], cmap='hot')
        axes[row, 3].set_title(f'HEATMAP GT\nMin: {annotations["HEATMAP"].min():.2f}\nMax: {annotations["HEATMAP"].max():.2f}')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
    else:
        axes[row, 3].text(0.5, 0.5, 'HEATMAP\nIndisponible', ha='center', va='center')
        axes[row, 3].set_title('HEATMAP GT')
    axes[row, 3].axis('off')
    
    # LIGNE 2: Prédictions
    row = 1
    
    # Prédiction sémantique
    if 'semantic_mask' in pred_results:
        im = axes[row, 0].imshow(pred_results['semantic_mask'], cmap=cmap, vmin=0, vmax=19)
        pred_classes = np.unique(pred_results['semantic_mask'])
        axes[row, 0].set_title(f'Prédiction Sémantique\nClasses: {pred_classes}')
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)
    else:
        axes[row, 0].text(0.5, 0.5, 'Prédiction\nSémantique\nIndisponible', ha='center', va='center')
        axes[row, 0].set_title('Prédiction Sémantique')
    axes[row, 0].axis('off')
    
    # Prédiction instances
    if 'instance_mask' in pred_results:
        im = axes[row, 1].imshow(pred_results['instance_mask'], cmap='nipy_spectral')
        pred_instances = len(np.unique(pred_results['instance_mask'])) - 1
        axes[row, 1].set_title(f'Prédiction Instances\n{pred_instances} instances')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
    else:
        axes[row, 1].text(0.5, 0.5, 'Prédiction\nInstances\nIndisponible', ha='center', va='center')
        axes[row, 1].set_title('Prédiction Instances')
    axes[row, 1].axis('off')
    
    # Heatmap prédiction
    if 'heatmap' in pred_results:
        heatmap_data = pred_results['heatmap']
        if len(heatmap_data.shape) > 2:
            heatmap_data = np.max(heatmap_data, axis=0)  # Max sur les canaux
        im = axes[row, 2].imshow(heatmap_data, cmap='hot')
        axes[row, 2].set_title(f'Heatmap Prédiction\nMin: {heatmap_data.min():.2f}\nMax: {heatmap_data.max():.2f}')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
    else:
        axes[row, 2].text(0.5, 0.5, 'Heatmap\nPrédiction\nIndisponible', ha='center', va='center')
        axes[row, 2].set_title('Heatmap Prédiction')
    axes[row, 2].axis('off')
    
    # Instances avec boîtes
    if 'semantic_mask' in pred_results:
        axes[row, 3].imshow(pred_results['semantic_mask'], cmap=cmap, alpha=0.7)
        
        # Ajouter les boîtes des instances détectées
        boxes_added = 0
        for inst in pred_results['individual_instances']:
            if 'bbox' in inst:
                x1, y1, x2, y2 = inst['bbox']
                w, h = x2 - x1, y2 - y1
                
                rect = patches.Rectangle((x1, y1), w, h, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', alpha=0.8)
                axes[row, 3].add_patch(rect)
                
                # Label avec confiance
                label = f"{inst['confidence']:.2f}"
                if 'predicted_class' in inst:
                    label += f"\nC{inst['predicted_class']}"
                
                axes[row, 3].text(x1, y1-2, label, 
                                fontsize=6, color='red', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                boxes_added += 1
        
        axes[row, 3].set_title(f'Prédictions + Boîtes\n{boxes_added} détections')
    else:
        axes[row, 3].text(0.5, 0.5, 'Prédictions\navec Boîtes\nIndisponible', ha='center', va='center')
        axes[row, 3].set_title('Prédictions + Boîtes')
    axes[row, 3].axis('off')
    
    # LIGNE 3: Comparaisons et analyses
    row = 2
    
    # Différence sémantique TARGET vs Prédiction
    if annotations.get('TARGET') is not None and 'semantic_mask' in pred_results:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]
        
        pred_sem = pred_results['semantic_mask']
        
        # Redimensionner si nécessaire
        if target_data.shape != pred_sem.shape:
            print(f"⚠️  Redimensionnement nécessaire: GT {target_data.shape} vs Pred {pred_sem.shape}")
            import cv2
            if target_data.size < pred_sem.size:
                target_data = cv2.resize(target_data, (pred_sem.shape[1], pred_sem.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                pred_sem = cv2.resize(pred_sem, (target_data.shape[1], target_data.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        diff_mask = (target_data != pred_sem).astype(float)
        accuracy = 1 - np.mean(diff_mask)
        
        im = axes[row, 0].imshow(diff_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Différence Sémantique\nAccuracy: {accuracy:.3f}')
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)
    else:
        axes[row, 0].text(0.5, 0.5, 'Comparaison\nSémantique\nImpossible', ha='center', va='center')
        axes[row, 0].set_title('Différence Sémantique')
    axes[row, 0].axis('off')
    
    # Comparaison des heatmaps
    if annotations.get('HEATMAP') is not None and 'heatmap' in pred_results:
        gt_heat = annotations['HEATMAP']
        pred_heat = pred_results['heatmap']
        
        if len(pred_heat.shape) > 2:
            pred_heat = np.max(pred_heat, axis=0)
        
        # Normaliser les deux heatmaps
        gt_heat_norm = (gt_heat - gt_heat.min()) / (gt_heat.max() - gt_heat.min() + 1e-8)
        pred_heat_norm = (pred_heat - pred_heat.min()) / (pred_heat.max() - pred_heat.min() + 1e-8)
        
        # Différence
        heat_diff = np.abs(gt_heat_norm - pred_heat_norm)
        mse = np.mean(heat_diff ** 2)
        
        im = axes[row, 1].imshow(heat_diff, cmap='viridis')
        axes[row, 1].set_title(f'Différence Heatmaps\nMSE: {mse:.4f}')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
    else:
        axes[row, 1].text(0.5, 0.5, 'Comparaison\nHeatmaps\nImpossible', ha='center', va='center')
        axes[row, 1].set_title('Différence Heatmaps')
    axes[row, 1].axis('off')
    
    # ZONES si disponible
    if annotations.get('ZONES') is not None:
        im = axes[row, 2].imshow(annotations['ZONES'], cmap='Set3')
        unique_zones = np.unique(annotations['ZONES'])
        axes[row, 2].set_title(f'ZONES GT\nZones: {unique_zones}')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
    else:
        axes[row, 2].text(0.5, 0.5, 'ZONES\nIndisponible', ha='center', va='center')
        axes[row, 2].set_title('ZONES GT')
    axes[row, 2].axis('off')
    
    # Superposition finale
    if annotations.get('PARCELID') is not None and 'instance_mask' in pred_results:
        # Créer superposition RGB
        H, W = annotations['PARCELID'].shape
        overlay = np.zeros((H, W, 3))
        
        # GT en rouge
        gt_mask = (annotations['PARCELID'] > 0).astype(float)
        overlay[:, :, 0] = gt_mask
        
        # Prédictions en vert  
        pred_mask = (pred_results['instance_mask'] > 0).astype(float)
        if pred_mask.shape != (H, W):
            import cv2
            pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        overlay[:, :, 1] = pred_mask
        
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Superposition Finale\nGT(rouge) + Pred(vert)\nOverlap(jaune)')
    else:
        axes[row, 3].text(0.5, 0.5, 'Superposition\nImpossible', ha='center', va='center')
        axes[row, 3].set_title('Superposition')
    axes[row, 3].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = f'complete_analysis_tile_{tile_id}_fold5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Analyse complète sauvegardée: {output_file}")
    
    plt.show()
    
    # Statistiques détaillées
    print_comprehensive_stats(annotations, pred_results, tile_id)

def print_comprehensive_stats(annotations, pred_results, tile_id):
    """Affiche des statistiques complètes"""
    
    print(f"\n📊 STATISTIQUES COMPLÈTES - Tuile {tile_id}")
    print("="*60)
    
    # Ground Truth
    print("🎯 GROUND TRUTH:")
    if annotations.get('PARCELID') is not None:
        n_parcels = len(np.unique(annotations['PARCELID'])) - 1
        print(f"   Parcelles: {n_parcels}")
    
    if annotations.get('TARGET') is not None:
        target_data = annotations['TARGET']
        if len(target_data.shape) == 3:
            target_data = target_data[0]
        unique_targets = np.unique(target_data)
        print(f"   Classes TARGET: {len(unique_targets)} ({unique_targets})")
    
    if annotations.get('INSTANCES') is not None:
        n_instances = len(np.unique(annotations['INSTANCES'])) - 1
        print(f"   Instances: {n_instances}")
    
    # Prédictions
    print("\n🔮 PRÉDICTIONS:")
    if 'individual_instances' in pred_results:
        n_pred = len(pred_results['individual_instances'])
        confidences = [inst['confidence'] for inst in pred_results['individual_instances']]
        if confidences:
            print(f"   Instances détectées: {n_pred}")
            print(f"   Confiance moyenne: {np.mean(confidences):.3f}")
            print(f"   Confiance min/max: {np.min(confidences):.3f}/{np.max(confidences):.3f}")
    
    if 'semantic_mask' in pred_results:
        pred_classes = np.unique(pred_results['semantic_mask'])
        print(f"   Classes prédites: {len(pred_classes)} ({pred_classes})")
    
    # Comparaisons possibles
    print("\n⚖️  COMPARAISONS:")
    if annotations.get('TARGET') is not None and 'semantic_mask' in pred_results:
        print("   ✅ Comparaison sémantique possible")
    else:
        print("   ❌ Comparaison sémantique impossible")
    
    if annotations.get('INSTANCES') is not None and 'instance_mask' in pred_results:
        print("   ✅ Comparaison instances possible")
    else:
        print("   ❌ Comparaison instances impossible")
    
    if annotations.get('HEATMAP') is not None and 'heatmap' in pred_results:
        print("   ✅ Comparaison heatmaps possible")
    else:
        print("   ❌ Comparaison heatmaps impossible")

# UTILISATION
def main():
    tile_id = "30000"
    inference_file = "preds/S2_30000_best_fold_5.npy"
    
    visualize_complete_comparison(tile_id, inference_file)

if __name__ == "__main__":
    main()