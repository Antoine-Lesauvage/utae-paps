"""
Main script for panoptic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pprint
import time

import numpy as np
import torch
import torch.utils.data as data_t
import torchnet as tnt

from src import model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.panoptic.FocalLoss import FocalLoss
from src.utils import pad_collate, get_ntrainparams
from src.utils import create_weighted_sampler
from src.temporal_augmentation import PASTIS_Dataset_WithAugmentation, create_vine_orchard_augmentation
parser = argparse.ArgumentParser()
# PaPs Parameters
## Architecture Hyperparameters
parser.add_argument("--shape_size", default=16, type=int, help="Shape size for PaPs")
parser.add_argument(
    "--no_mask_conv",
    dest="mask_conv",
    action="store_false",
    help="With this flag no residual CNN is used after combination of global saliency and local shape.",
)
parser.add_argument(
    "--backbone",
    default="utae",
    type=str,
    help="Backbone encoder for PaPs (utae or uconvlstm)",
)

## Losses & metrics
parser.add_argument(
    "--l_center", default=1, type=float, help="Coefficient for centerness loss"
)
parser.add_argument("--l_size", default=1, type=float, help="Coefficient for size loss")
parser.add_argument(
    "--l_shape", default=1, type=float, help="Coefficient for shape loss"
)
parser.add_argument(
    "--l_class", default=1, type=float, help="Coefficient for class loss"
)
parser.add_argument(
    "--beta", default=4, type=float, help="Beta parameter for centerness loss"
)
parser.add_argument(
    "--no_autotune",
    dest="autotune",
    action="store_false",
    help="If this flag is used the confidence threshold for the pseudo-nms will NOT be tuned automatically on the validation set",
)
parser.add_argument(
    "--no_supmax",
    dest="supmax",
    action="store_false",
    help="If this flag is used, ALL local maxima are supervised (and not just the more confident center per ground truth object)",
)
parser.add_argument(
    "--warmup",
    default=5,
    type=int,
    help="Number of epochs to do with only the centerness loss as supervision.",
)
parser.add_argument(
    "--val_metrics_only",
    dest="val_metrics_only",
    action="store_true",
    help="If true, panoptic metrics are computed only on validation and test epochs.",
)
parser.add_argument(
    "--val_every",
    default=5,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=65,
    type=int,
    help="Do validation only after that many epochs.",
)

## Thresholds
parser.add_argument(
    "--min_remain",
    default=0.5,
    type=float,
    help="Minimum remain fraction for the pseudo-nms.",
)
parser.add_argument(
    "--mask_threshold",
    default=0.4,
    type=float,
    help="Binary threshold for instancce masks.",
)
parser.add_argument(
    "--min_confidence",
    default=0.2,
    type=float,
    help="Minimum confidence for the pseudo-nms (tuned automatically by default)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

## Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument(
    "--mono_date",
    default=None,
    type=str,
    help="If one date is specified, model is training on a single date.",
)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--void_label", default=19, type=int)
parser.add_argument("--background_label", default=0, type=int)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--use_focal_loss",
    dest="use_focal_loss",
    action="store_true",
    help="Use Focal Loss instead of standard PaPs loss"
)
parser.add_argument(
    "--focal_gamma", 
    default=2.0, 
    type=float, 
    help="Gamma parameter for Focal Loss (focus on hard examples)"
)
parser.add_argument(
    "--focal_alpha", 
    default="0.1, 3.0, 15.0, 0.8",  # Sera calculé automatiquement
    type=str, 
    help="Alpha parameter for Focal Loss (class weighting)"
)

parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
parser.add_argument(
    "--temporal_aug",
    dest="temporal_aug",
    action="store_true",
    help="Enable temporal augmentations"
)
parser.add_argument(
    "--aug_factor", 
    default=2, 
    type=int, 
    help="Temporal augmentation factor"
)
list_args = ["encoder_widths", "decoder_widths", "out_conv"]

parser.set_defaults(
    cache=False, mask_conv=True, supmax=True, autotune=True, val_metrics_only=False
)

def parse_focal_alpha(alpha_str, num_classes=4):
    """
    Parse alpha weights from comma-separated string
    """
    if alpha_str is None:
        return None
    
    try:
        # Parser les valeurs séparées par des virgules
        alpha_values = [float(x.strip()) for x in alpha_str.split(',')]
        
        if len(alpha_values) != num_classes:
            raise ValueError(f"Expected {num_classes} alpha values, got {len(alpha_values)}")
        
        print(f"✅ Poids Alpha parsés: {alpha_values}")
        return torch.tensor(alpha_values, dtype=torch.float32)
    
    except Exception as e:
        print(f"❌ Erreur parsing alpha weights: {e}")
        print(f"Format attendu: 'val1,val2,val3,val4' (exemple: '1.16,29.78,250.42,10.11')")
        return None


def iterate(
    model,
    data_loader,
    criterion,
    config,
    optimizer=None,
    mode="train",
    device=None,
    compute_metrics=True,
    heatmap_only=False,
    autotune=False,
):
    loss_meter = tnt.meter.AverageValueMeter()
    loss_center_meter = tnt.meter.AverageValueMeter()
    loss_size_meter = tnt.meter.AverageValueMeter()
    loss_shape_meter = tnt.meter.AverageValueMeter()
    loss_class_meter = tnt.meter.AverageValueMeter()

    if compute_metrics:
        pano_meter = PanopticMeter(
            num_classes=config.num_classes, void_label=config.void_label
        )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch

        if mode != "train":
            with torch.no_grad():
                predictions = model(
                    x,
                    batch_positions=dates,
                    pseudo_nms=compute_metrics,
                    heatmap_only=heatmap_only,
                )
        else:
            zones = y[:, :, :, 2] if config.supmax else None
            optimizer.zero_grad()
            predictions = model(
                x,
                batch_positions=dates,
                pseudo_nms=compute_metrics,
                zones=zones,
                heatmap_only=heatmap_only,
            )
        loss = criterion(predictions, y, heatmap_only=heatmap_only)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if compute_metrics:
            pano_meter.add(predictions, y)

        ce, si, sh, cl = criterion.value
        loss_center_meter.add(ce)
        loss_size_meter.add(si)
        loss_shape_meter.add(sh)
        loss_class_meter.add(cl)
        loss_meter.add(float(loss.item()))

        if (i + 1) % config.display_step == 0:
            if compute_metrics:
                sq, rq, pq = pano_meter.value()
                print(
                    "Step [{}/{}], Loss: {:.4f}, SQ {:.1f},  RQ {:.1f}  , PQ {:.1f}".format(
                        i + 1,
                        len(data_loader),
                        loss_meter.value()[0],
                        sq * 100,
                        rq * 100,
                        pq * 100,
                    )
                )
            else:
                print(
                    "Step [{}/{}], Loss: {:.4f} ".format(
                        i + 1, len(data_loader), loss_meter.value()[0]
                    )
                )

    if autotune:
        thrsh = tune_threshold(criterion.predicted_confidences, criterion.achieved_ious)
        model.min_confidence = torch.tensor(
            [thrsh], device=next(model.parameters()).device
        )
        config.min_confidence = thrsh

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))

    metrics = {
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_center_loss".format(mode): loss_center_meter.value()[0],
        "{}_size_loss".format(mode): loss_size_meter.value()[0],
        "{}_shape_loss".format(mode): loss_shape_meter.value()[0],
        "{}_class_loss".format(mode): loss_class_meter.value()[0],
        "{}_epoch_time".format(mode): total_time,
    }
    if compute_metrics:
        SQ, RQ, PQ = pano_meter.value()
        metrics.update(
            {
                "{}_SQ".format(mode): float(SQ),
                "{}_RQ".format(mode): float(RQ),
                "{}_PQ".format(mode): float(PQ),
            }
        )
    if mode == "test":
        return metrics, pano_meter.get_table()
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, table, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    np.save(os.path.join(config.res_dir, "Fold_{}".format(fold), "test_tables"), table)


def tune_threshold(confidences, target):
    t = target.squeeze()
    t = t > 0.5
    p = confidences.squeeze()
    best_score = 0
    best_threshold = 0
    print("Tuning confidence threshold . . . ")
    for ct in np.arange(0, 1, 0.01):
        TP = ((p > ct) * t).sum()
        FP = ((p > ct) * (~t)).sum()
        FN = ((p < ct) * t).sum()
        score = TP / (TP + 0.5 * (FP + FN))
        if score > best_score:
            best_threshold = ct
            best_score = score

    print(
        "Best F-Score : {:.2f} / Threshold : {:.2f}".format(best_score, best_threshold)
    )
    return best_threshold


def main(config):
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)

    device = torch.device(config.device)

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1  # Quick fix to launch different folds simultaneously
                # Configuration des arguments du dataset
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance",
        )
        if hasattr(config, 'class_mapping') and config.class_mapping is not None:
            dt_args['class_mapping'] = config.class_mapping
            print(f"✅ Class mapping ajouté aux arguments du dataset: {config.class_mapping}")
        
        # 🚀 DATASET D'ENTRAÎNEMENT AVEC AUGMENTATIONS TEMPORELLES
        print("🔄 Création du dataset d'entraînement avec augmentations temporelles...")
        temporal_aug_pipeline = create_vine_orchard_augmentation()
        
        dt_train = PASTIS_Dataset_WithAugmentation(
            **dt_args, 
            cache=config.cache, 
            folds=train_folds,
            temporal_aug_pipeline=temporal_aug_pipeline,
            augmentation_factors={
                0: 1,    # Background: pas d'augmentation
                1: 6,    # Vigne: x6 (compenser rareté)
                2: 12,   # Verger: x12 (compenser extrême rareté)
                3: 3     # Vide: x3
            },
            target_classes=[1, 2, 3],  # Classes à favoriser
        )
        
        # 📊 DATASETS DE VALIDATION/TEST SANS AUGMENTATIONS
        dt_val = PASTIS_Dataset(**dt_args, folds=val_fold)
        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)
        
        # Vérification de la configuration
        print("🔍 VÉRIFICATION DU DATASET")
        print(f"Dataset train - classe: {type(dt_train).__name__}")
        print(f"Dataset val - classe: {type(dt_val).__name__}")
        print(f"Dataset test - classe: {type(dt_test).__name__}")
        print(f"Train: {len(dt_train)} échantillons (avec augmentations)")
        print(f"Val: {len(dt_val)} échantillons")
        print(f"Test: {len(dt_test)} échantillons")
        
        # 🎯 CRÉATION DU SAMPLER ÉQUILIBRÉ
        print("⚖️ Création du sampler équilibré pour l'entraînement...")
        train_sampler = create_weighted_sampler(
            dt_train,
                class_weights={
        0: 0.2,   # Réduire Background
        1: 0.4,   # Augmenter Vigne
        2: 0.35,  # Augmenter Verger (classe la plus rare)  
        3: 0.05   # Vide reste minoritaire
    }
        )
        
        # 🔄 DATALOADERS
        train_loader = data_t.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            sampler=train_sampler,  # ← Utiliser le sampler au lieu de shuffle
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=config.num_workers,
        )
        
        # Validation et test sans sampler (distribution naturelle)
        val_loader = data_t.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=config.num_workers,
        )
        
        test_loader = data_t.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=config.num_workers,
        )

        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )

        #model = model_utils.get_model(config, mode="panoptic")
        if getattr(config, 'use_vine_orchard_specialization', False):
            print("🍇 Utilisation du modèle spécialisé vignes/vergers")
            model = model_utils.get_vine_orchard_model(config)
        else:
            print("📊 Utilisation du modèle standard")
            model = model_utils.get_model(config, mode="panoptic")

        config.N_params = get_ntrainparams(model)
        #with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            #file.write(json.dumps(vars(config), indent=4))
        fold_config_path = os.path.join(config.res_dir, f"Fold_{fold + 1}", "config.json")
        os.makedirs(os.path.dirname(fold_config_path), exist_ok=True)
        with open(fold_config_path, "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        print(f"💾 Configuration sauvegardée : {fold_config_path}")
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR( #AL
        #    optimizer=optimizer, milestones=[60, 80], gamma=0.3
        #)
        # Par un scheduler plus adapté au transfert learning :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        model.apply(weight_init)
        # GESTION DE LA REPRISE D'ENTRAÎNEMENT
        start_epoch = 0
        trainlog = {}
        best_pq = -1.0

        # Vérifier s'il faut reprendre depuis un checkpoint
        if hasattr(config, 'resume_checkpoint') and config.resume_checkpoint:
            print(f"🔄 Reprise d'entraînement depuis : {config.resume_checkpoint}")
    
            try:
                # Charger le checkpoint
                resume_data = torch.load(config.resume_checkpoint, map_location=device)
        
                # Restaurer l'état du modèle et de l'optimiseur
                model.load_state_dict(resume_data['state_dict'])
                if 'optimizer' in resume_data:
                    optimizer.load_state_dict(resume_data['optimizer'])
        
                start_epoch = resume_data.get('epoch', 0)
        
                # Charger le log d'entraînement existant
                log_path = os.path.join(config.res_dir, f"Fold_{fold + 1}", "trainlog.json")
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        trainlog = json.load(f)
            
                    # Trouver le meilleur PQ précédent
                    for epoch_log in trainlog.values():
                        if 'val_PQ' in epoch_log:
                            best_pq = max(best_pq, epoch_log['val_PQ'])
        
                print(f"✅ Reprise réussie depuis l'époque {start_epoch}")
                print(f"📊 Meilleur PQ précédent : {best_pq:.4f}")
        
            except Exception as e:
                print(f"❌ Erreur lors de la reprise : {e}")
                print("🔄 Démarrage d'un nouvel entraînement")
                start_epoch = 0
                trainlog = {}
                best_pq = -1.0
                

        # Dans la fonction main(), remplacer la section criterion par :
        if getattr(config, 'use_focal_loss', False):
            print("🎯 Initialisation de Focal Loss pour classes déséquilibrées")
            alpha_weights = parse_focal_alpha(config.focal_alpha, num_classes=4)
            # Créer une loss combinée : Focal Loss + PaPs Loss
            focal_criterion = FocalLoss(
        alpha=alpha_weights,
        gamma=config.focal_gamma,
        ignore_label=config.void_label
            )
    
            # Loss hybride pour la segmentation panoptique
            class HybridPanopticLoss(torch.nn.Module):
                def __init__(self, focal_loss, paps_loss, focal_weight=0.5):
                    super().__init__()
                    self.focal_loss = focal_loss
                    self.paps_loss = paps_loss
                    self.focal_weight = focal_weight
                    self.value = [0.0, 0.0, 0.0, 0.0]  # [center, size, shape, class]
        
                def forward(self, predictions, targets, heatmap_only=False):
        # Loss PaPs standard
                    paps_loss_value = self.paps_loss(predictions, targets, heatmap_only)
        
        # Toujours synchroniser les attributs PaPs
                    self.value = self.paps_loss.value
                    if hasattr(self.paps_loss, 'predicted_confidences'):
                        self.predicted_confidences = self.paps_loss.predicted_confidences
                    if hasattr(self.paps_loss, 'achieved_ious'):
                        self.achieved_ious = self.paps_loss.achieved_ious
        
                    if not heatmap_only and "semantic" in predictions:
                        try:
                            semantic_pred = predictions["semantic"]
                            device = semantic_pred.device
                
                            if semantic_pred.dim() == 2:
                    # Cas prédictions déjà flattened
                                semantic_pred_flat = semantic_pred
                    
                    # Extraire targets sémantiques
                                if targets.dim() == 4:
                                    semantic_targets = targets[:, :, :, 6].long()
                                else:
                                    semantic_targets = targets.long()
                    
                    # Flatten les targets
                                semantic_targets_flat = semantic_targets.view(-1)
                    
                    # Ajuster la taille si nécessaire
                                if semantic_pred_flat.shape[0] != semantic_targets_flat.shape[0]:
                                    N = semantic_pred_flat.shape[0]
                                    semantic_targets_flat = semantic_targets_flat[:N]
                    
                    # S'assurer que tout est sur le même device
                                semantic_targets_flat = semantic_targets_flat.to(device)
                    
                    # S'assurer que alpha est sur le bon device
                                if hasattr(self.focal_loss, 'alpha') and self.focal_loss.alpha is not None:
                                    self.focal_loss.alpha = self.focal_loss.alpha.to(device)
                    
                    # Vérifications de sécurité
                                max_class = semantic_targets_flat.max().item()
                                if max_class >= semantic_pred_flat.shape[1]:
                                    return paps_loss_value
                    
                    # Appliquer Focal Loss
                                focal_loss_value = self.focal_loss(semantic_pred_flat, semantic_targets_flat)
                    
                    # Combiner les losses
                                total_loss = (1 - self.focal_weight) * paps_loss_value + self.focal_weight * focal_loss_value
                    
                    # Mettre à jour les métriques (optionnel)
                                paps_values = list(self.paps_loss.value)
                                if len(paps_values) >= 4:
                                    paps_values[3] = focal_loss_value.item()  # Remplacer class loss
                                    self.value = paps_values
                    
                                #print(f"📊 PaPs: {paps_loss_value.item():.4f}, Focal: {focal_loss_value.item():.4f}, Total: {total_loss.item():.4f}")
                    
                                return total_loss
                            else:
                                return paps_loss_value
                    
                        except Exception as e:
                            print(f"⚠️ Focal Loss failed, using PaPs only: {e}")
                            return paps_loss_value
        
                    return paps_loss_value
    
    # Créer la loss PaPs standard
            paps_criterion = PaPsLoss(
        l_center=config.l_center,
        l_size=config.l_size,
        l_shape=config.l_shape,
        l_class=config.l_class,
        beta=config.beta,
        void_label=config.void_label,
            )
    
    # Combiner les deux
            criterion = HybridPanopticLoss(focal_criterion, paps_criterion, focal_weight=1.0)
    
            print(f"✅ Loss hybride configurée : gamma={config.focal_gamma}, poids focal={1.0}")
    
        else:
    # Loss PaPs standard
            criterion = PaPsLoss(
        l_center=config.l_center,
        l_size=config.l_size,
        l_shape=config.l_shape,
        l_class=config.l_class,
        beta=config.beta,
        void_label=config.void_label,
            )
            print("📊 Utilisation de la loss PaPs standard")

        best_pq = -1.0
        for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
            print("EPOCH {}/{}".format(epoch, start_epoch + config.epochs))
            if epoch % 5 == 0 or epoch == config.epochs:  # Sauvegarder tous les 5 époques
                torch.save(
            {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
                },
                os.path.join(
                config.res_dir, "Fold_{}".format(fold + 1), f"model_epoch_{epoch}.pth.tar"
                ),
                    )
            heatmap_only = epoch - 1 < config.warmup if config.warmup > 0 else False
            model.train()
            train_metrics = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
                compute_metrics=epoch > config.warmup and not config.val_metrics_only,
                heatmap_only=heatmap_only,
            )

            trainlog[epoch] = {**train_metrics}
            scheduler.step()
            if (
                epoch > config.warmup
                and epoch % config.val_every == 0
                and epoch > config.val_after
            ):
                print("Validation . . . ")
                model.eval()
                val_metrics = iterate(
                    model,
                    data_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                    compute_metrics=epoch > config.warmup,
                    heatmap_only=heatmap_only,
                    autotune=config.autotune,
                )
                trainlog[epoch].update(val_metrics)
                val_pq = val_metrics["val_PQ"]
                if val_pq > best_pq:
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                        ),
                    )
                    best_pq = val_pq
            checkpoint(fold + 1, trainlog, config)

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()
        test_metrics, tables = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,
            mode="test",
            device=device,
        )
        save_results(fold + 1, test_metrics, tables, config)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    pprint.pprint(config)
    main(config)
    
