#!/usr/bin/env python3
"""
Script d'entraînement pour la spécialisation vignes/vergers
Utilise le transfert learning depuis le modèle UTAE+PAPS pré-entraîné
"""
import argparse
import json
import os
import pprint

# Import du script d'entraînement principal
from train_panoptic import main, parser, list_args

def setup_vine_orchard_config():
    """Configure les paramètres pour l'entraînement vignes/vergers"""
    
    # Arguments spécifiques au transfert learning
    parser.add_argument(
        "--pretrained_weights", 
        type=str, 
        help="Chemin vers les poids pré-entraînés (.pth.tar)"
    )
    parser.add_argument(
        "--pretrained_fold", 
        default=1, 
        type=int, 
        help="Fold à utiliser pour les poids pré-entraînés (1-5)"
    )
    parser.add_argument(
        "--freeze_encoder", 
        action="store_true", 
        help="Geler l'encodeur pendant l'entraînement"
    )
    parser.add_argument(
        "--vine_orchard_lr", 
        default=0.001, 
        type=float, 
        help="Learning rate pour le fine-tuning"
    )
    
    # Parse arguments
    config = parser.parse_args()
    
    # Parser les listes comme dans le script original
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            if isinstance(v, str):
                v = v.replace("[", "")
                v = v.replace("]", "")
                config.__setattr__(k, list(map(int, v.split(","))))
    
    # Configuration spécifique vignes/vergers
    config.use_vine_orchard_specialization = True
    config.num_classes = 4
    config.background_label = 0
    config.void_label = 3  # Le label vide
    config.out_conv = [32, 4]

    
    # Learning rate adapté pour le transfert learning
    if hasattr(config, 'vine_orchard_lr'):
        config.lr = config.vine_orchard_lr
    
    # Chemin vers les poids pré-entraînés
    if config.pretrained_weights is None:
        weights_dir = "/home/onyxia/work/UTAE_PAPs"
        config.pretrained_weights = os.path.join(
            weights_dir, 
            f"Fold_{config.pretrained_fold}", 
            "model.pth.tar"
        )
    
    # Ajouter le mapping de classes
    from src.utils import VINE_ORCHARD_CLASS_MAPPING
    config.class_mapping = VINE_ORCHARD_CLASS_MAPPING

# Ajouter une vérification :
    print(f"🎯 Mapping ajouté à la config: {config.class_mapping}")
    
    print(f"🔢 Type du mapping: {type(config.class_mapping)}")
    
    print("🍇 CONFIGURATION VIGNES/VERGERS")
    print("=" * 50)
    print(f"📁 Poids pré-entraînés : {config.pretrained_weights}")
    print(f"🎯 Nombre de classes : {config.num_classes}")
    print(f"📚 Learning rate : {config.lr}")
    print(f"🔒 Encodeur gelé : {getattr(config, 'freeze_encoder', False)}")
    print(f"📊 Mapping de classes actif")
    print("=" * 50)
    
    return config

if __name__ == "__main__":
    config = setup_vine_orchard_config()
    pprint.pprint(vars(config))
    main(config)