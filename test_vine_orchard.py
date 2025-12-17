#!/usr/bin/env python3
"""
Script de test pour les modèles vignes/vergers entraînés sur les 5 folds
Basé sur test_panoptic.py mais adapté pour 4 classes
"""
import argparse
import json
import os
import pprint
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data_t

from src import model_utils
from src.dataset import PASTIS_Dataset
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.panoptic.FocalLoss import FocalLoss
from src.utils import pad_collate, VINE_ORCHARD_CLASS_MAPPING

def load_model_config(model_path):
    """Charge la configuration depuis le fichier config.json du fold"""
    fold_dir = os.path.dirname(model_path)
    config_path = os.path.join(fold_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convertir en objet de configuration
        from types import SimpleNamespace
        config = SimpleNamespace(**config_dict)
        return config
    else:
        print(f"⚠️ Config non trouvée : {config_path}")
        return None

def test_single_fold(fold, model_path, dataset_folder, device='cuda'):
    """Teste un modèle sur un fold spécifique"""
    
    print(f"\n🧪 TEST FOLD {fold}")
    print("=" * 50)
    
    # Charger la configuration du modèle
    config = load_model_config(model_path)
    if not config:
        print(f"❌ Impossible de charger la config pour fold {fold}")
        return None
    
    print(f"📁 Modèle : {model_path}")
    print(f"🎯 Classes : {config.num_classes}")
    print(f"📊 Void label : {config.void_label}")
    
    # Créer le dataset de test
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]
    
    test_fold = fold_sequence[fold - 1][2]  # Fold de test
    
    dt_args = dict(
        folder=dataset_folder,
        norm=True,
        reference_date=config.ref_date,
        mono_date=getattr(config, 'mono_date', None),
        target="instance",
        class_mapping=VINE_ORCHARD_CLASS_MAPPING,
    )
    
    dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)
    
    test_loader = data_t.DataLoader(
        dt_test,
        batch_size=getattr(config, 'batch_size', 4),
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
        num_workers=0,
    )
    
    print(f"📊 Dataset test : {len(dt_test)} échantillons")
    
    # Créer le modèle
    model = model_utils.get_vine_orchard_model(config)
    
    # Charger les poids
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"✅ Poids chargés (époque {epoch})")
    except Exception as e:
        print(f"❌ Erreur chargement poids : {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Métriques
    pano_meter = PanopticMeter(
        num_classes=config.num_classes, 
        void_label=config.void_label
    )
    
    criterion = FocalLoss(
        gamma=config.focal_gamma,
        alpha=config.focal_alpha,
        ignore_label=config.ignore_index,
        void_label=config.void_label
    )
    
    # Test
    print("🔬 Évaluation en cours...")
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Déplacer sur device
            if device != 'cpu':
                batch = recursive_todevice(batch, device)
            
            (x, dates), y = batch
            
            # Prédiction
            predictions = model(
                x,
                batch_positions=dates,
                pseudo_nms=True,
            )
            
            # Loss
            loss = criterion(predictions, y)
            total_loss += loss.item()
            num_batches += 1
            
            # Métriques
            pano_meter.add(predictions, y)
            
            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(test_loader)}")
    
    # Résultats
    SQ, RQ, PQ = pano_meter.value()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    results = {
        'fold': fold,
        'test_loss': avg_loss,
        'test_SQ': float(SQ),
        'test_RQ': float(RQ),
        'test_PQ': float(PQ),
        'num_samples': len(dt_test),
    }
    
    print(f"📈 RÉSULTATS FOLD {fold}:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   SQ: {SQ:.1%}")
    print(f"   RQ: {RQ:.1%}")
    print(f"   PQ: {PQ:.1%}")
    
    # Table détaillée par classe
    table = pano_meter.get_table()
    results['detailed_table'] = table
    
    return results

def recursive_todevice(x, device):
    """Déplace récursivement les tenseurs sur le device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def test_all_folds(results_dir, dataset_folder, device='cuda'):
    """Teste tous les folds et compile les résultats"""
    
    print("🧪 TEST COMPLET - MODÈLES VIGNES/VERGERS")
    print("=" * 60)
    
    all_results = []
    
    # Tester chaque fold
    for fold in range(1, 6):
        model_path = os.path.join(results_dir, f"Fold_{fold}", "model.pth.tar")
        
        if not os.path.exists(model_path):
            print(f"❌ Modèle non trouvé : {model_path}")
            continue
        
        try:
            results = test_single_fold(fold, model_path, dataset_folder, device)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"❌ Erreur fold {fold}: {e}")
            continue
    
    if not all_results:
        print("❌ Aucun résultat obtenu")
        return
    
    # Statistiques agrégées
    print("\n📊 RÉSULTATS AGRÉGÉS")
    print("=" * 60)
    
    metrics = ['test_loss', 'test_SQ', 'test_RQ', 'test_PQ']
    stats = defaultdict(list)
    
    for result in all_results:
        for metric in metrics:
            stats[metric].append(result[metric])
    
    print(f"{'Métrique':<15} {'Moyenne':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for metric in metrics:
        values = stats[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if 'loss' in metric:
            print(f"{metric:<15} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
        else:
            print(f"{metric:<15} {mean_val:<10.1%} {std_val:<10.1%} {min_val:<10.1%} {max_val:<10.1%}")
    
    # Sauvegarder les résultats
    output_file = os.path.join(results_dir, "test_results_summary.json")
    summary = {
        'individual_folds': all_results,
        'statistics': {
            metric: {
                'mean': float(np.mean(stats[metric])),
                'std': float(np.std(stats[metric])),
                'min': float(np.min(stats[metric])),
                'max': float(np.max(stats[metric])),
            }
            for metric in metrics
        },
        'total_samples': sum(r['num_samples'] for r in all_results),
        'successful_folds': len(all_results),
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n💾 Résultats sauvegardés : {output_file}")
    
    # Affichage final
    print(f"\n🎯 BILAN FINAL :")
    print(f"   Folds testés: {len(all_results)}/5")
    print(f"   Échantillons totaux: {summary['total_samples']}")
    print(f"   PQ moyen: {summary['statistics']['test_PQ']['mean']:.1%} ± {summary['statistics']['test_PQ']['std']:.1%}")
    print(f"   SQ moyen: {summary['statistics']['test_SQ']['mean']:.1%} ± {summary['statistics']['test_SQ']['std']:.1%}")
    print(f"   RQ moyen: {summary['statistics']['test_RQ']['mean']:.1%} ± {summary['statistics']['test_RQ']['std']:.1%}")

def main():
    parser = argparse.ArgumentParser(description="Test des modèles vignes/vergers")
    parser.add_argument("--results_dir", required=True, help="Répertoire contenant les résultats des 5 folds")
    parser.add_argument("--dataset_folder", required=True, help="Chemin vers le dataset PASTIS")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fold", type=int, help="Tester un fold spécifique (1-5)")
    
    args = parser.parse_args()
    
    if args.fold:
        # Test d'un fold spécifique
        model_path = os.path.join(args.results_dir, f"Fold_{args.fold}", "model.pth.tar")
        test_single_fold(args.fold, model_path, args.dataset_folder, args.device)
    else:
        # Test de tous les folds
        test_all_folds(args.results_dir, args.dataset_folder, args.device)

if __name__ == "__main__":
    main()