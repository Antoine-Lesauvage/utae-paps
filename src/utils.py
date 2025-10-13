import collections.abc
import re

import torch
from torch.nn import functional as F
from torch.utils import data

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    if padlen <= 0:
        return x
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # Version robuste avec gestion d'erreurs
    elem = batch[0]
    elem_type = type(elem)
    
    try:
        if isinstance(elem, torch.Tensor):
            if len(elem.shape) > 0:
                sizes = [e.shape[0] for e in batch]
                m = max(sizes)
                if not all(s == m for s in sizes):
                    # Pad tensors avec vérification
                    padded_batch = []
                    for e in batch:
                        padded = pad_tensor(e, m, pad_value=pad_value)
                        padded_batch.append(padded)
                    batch = padded_batch
            
            # Stack sans paramètre out pour éviter les warnings
            return torch.stack(batch, dim=0)
            
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
                # Gérer les arrays numpy
                return pad_collate([torch.as_tensor(b) for b in batch], pad_value)
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
                
        elif isinstance(elem, collections.abc.Mapping):
            return {key: pad_collate([d[key] for d in batch], pad_value) for key in elem}
            
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(pad_collate(samples, pad_value) for samples in zip(*batch)))
            
        elif isinstance(elem, collections.abc.Sequence):
            # Vérifier la cohérence de taille
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("Inconsistent batch element sizes")
            transposed = zip(*batch)
            return [pad_collate(samples, pad_value) for samples in transposed]
    
    except Exception as e:
        print(f"⚠️ Erreur dans pad_collate: {e}")
        print(f"Element type: {elem_type}, Batch size: {len(batch)}")
        # Fallback : retourner le premier élément
        return batch[0] if batch else None

    raise TypeError(f"Format not managed: {elem_type}")

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# TRANSFERT LEARNING VIGNES/VERGERS - Mapping des classes
# =============================================================================

# Remplacer le mapping existant par :
VINE_ORCHARD_CLASS_MAPPING = {
    0: 0,   # Arrière-plan -> Background
    1: 0,   # Prairie -> Background
    2: 0,   # Blé tendre d'hiver -> Background
    3: 0,   # Maïs -> Background
    4: 0,   # Orge d'hiver -> Background
    5: 0,   # Colza d'hiver -> Background
    6: 0,   # Orge de printemps -> Background
    7: 0,   # Tournesol -> Background
    8: 1,   # Vigne -> Vigne ✓
    9: 0,   # Betterave -> Background
    10: 0,  # Triticale d'hiver -> Background
    11: 0,  # Blé dur d'hiver -> Background
    12: 0,  # Fruits, légumes, fleurs -> Background
    13: 0,  # Pommes de terre -> Background
    14: 0,  # Fourrage légumineux -> Background
    15: 0,  # Soja -> Background
    16: 2,  # Verger -> Verger ✓
    17: 0,  # Céréales mélangées -> Background
    18: 0,  # Sorgho -> Background
    19: 3,  # Label vide -> Vide ✓
}


# Noms des nouvelles classes
VINE_ORCHARD_CLASS_NAMES = {
    0: 'Background',
    1: 'Vigne', 
    2: 'Verger',
    3: 'Vide'
}

# Constantes pour la nouvelle configuration
NUM_VINE_ORCHARD_CLASSES = 4
VINE_ORCHARD_VOID_LABEL = 3  # Le label "vide" reste "vide"
VINE_ORCHARD_BACKGROUND_LABEL = 0

def adapt_model_for_vine_orchard(model, pretrained_weights_path=None):
    """
    Adapte un modèle UTAE+PAPS pré-entraîné (20 classes) pour la spécialisation vignes/vergers (3 classes).
    
    Args:
        model: Modèle UTAE+PAPS avec 3 classes
        pretrained_weights_path: Chemin vers les poids pré-entraînés (.pth.tar)
    
    Returns:
        model: Modèle adapté avec 3 classes
    """
    # 1. Charger les poids pré-entraînés si fournis
    if pretrained_weights_path:
        print(f"🔄 Chargement des poids pré-entraînés depuis {pretrained_weights_path}")
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        pretrained_state_dict = checkpoint['state_dict']
        
        # 2. Identifier les couches à adapter (celles avec num_classes=20)
        layers_to_adapt = []
        
        # Dans UTAE : encoder.out_conv (dernière couche)
        if 'encoder.out_conv.conv.conv.3.weight' in pretrained_state_dict:
            layers_to_adapt.append('encoder.out_conv.conv.conv.3')
            
        # Dans PaPs : class_mlp (dernière couche de classification)
        if 'class_mlp.4.weight' in pretrained_state_dict:
            layers_to_adapt.append('class_mlp.4')
            
        print(f"🎯 Couches à adapter détectées : {layers_to_adapt}")
        
        # 3. Charger tous les poids compatibles (encodeur, décodeur, etc.)
        model_state_dict = model.state_dict()
        compatible_weights = {}
        
        for name, param in pretrained_state_dict.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    compatible_weights[name] = param
                    print(f"✅ {name}: {param.shape}")
                else:
                    print(f"❌ {name}: forme incompatible {param.shape} vs {model_state_dict[name].shape}")
            else:
                print(f"⚠️  {name}: couche non trouvée dans le nouveau modèle")
        
        # 4. Charger les poids compatibles
        model.load_state_dict(compatible_weights, strict=False)
        print(f"🔄 {len(compatible_weights)} couches chargées avec succès")
        
        # 5. Initialiser les nouvelles couches finales (seulement Conv2d et Linear)
        print("🆕 Initialisation des nouvelles couches finales...")
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in ['out_conv', 'class_mlp']):
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)
                    print(f"🎲 Initialisé : {name}")
                    if hasattr(module, 'bias') and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.BatchNorm1d)):
                    # Les couches de normalisation sont initialisées automatiquement
                    print(f"🔧 Couche de normalisation : {name}")
    
    print("✅ Modèle adapté avec succès pour vignes/vergers (3 classes)")
    return model

def freeze_encoder_layers(model, freeze=True):
    """
    Gèle ou dégèle les couches de l'encodeur pour le transfert learning
    
    Args:
        model: Modèle UTAE+PAPS
        freeze: Si True, gèle l'encodeur. Si False, dégèle.
    """
    encoder_layers = [
        'encoder.in_conv',
        'encoder.down_blocks', 
        'encoder.temporal_encoder',
        'encoder.temporal_aggregator',
        'encoder.up_blocks'
    ]
    
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        
        # Vérifier si le paramètre appartient à l'encodeur
        is_encoder = any(layer in name for layer in encoder_layers)
        
        if is_encoder and freeze:
            param.requires_grad = False
            frozen_params += 1
            print(f"🔒 Gelé: {name}")
        elif is_encoder and not freeze:
            param.requires_grad = True
            print(f"🔓 Dégelé: {name}")
    
    if freeze:
        print(f"❄️  Encodeur gelé: {frozen_params}/{total_params} paramètres")
    else:
        print(f"🔥 Encodeur dégelé: tous les paramètres sont entraînables")
    
    return model


def analyze_dataset_classes(dataset, max_samples=1000):
    """
    Analyse la distribution des classes dans le dataset
    """
    import torch
    from collections import Counter
    
    print("🔍 Analyse de la distribution des classes...")
    
    class_counts = Counter()
    total_pixels = 0
    
    # Analyser un échantillon du dataset
    sample_size = min(len(dataset), max_samples)
    
    for i in range(0, sample_size, max(1, sample_size//100)):  # 100 échantillons max
        try:
            (data, dates), target = dataset[i]
            
            # Analyser les labels sémantiques (colonne 6)
            semantic_labels = target[:, :, 6].flatten()
            unique, counts = torch.unique(semantic_labels, return_counts=True)
            
            for cls, count in zip(unique.tolist(), counts.tolist()):
                class_counts[cls] += count
                
            total_pixels += semantic_labels.numel()
            
            if (i // max(1, sample_size//100)) % 10 == 0:
                print(f"   Échantillons analysés: {i+1}/{sample_size}")
                
        except Exception as e:
            print(f"Erreur échantillon {i}: {e}")
            continue
    
    # Calculer les poids
    print(f"\n📊 DISTRIBUTION DES CLASSES :")
    print(f"Total pixels analysés: {total_pixels:,}")
    
    weights = {}
    for cls in [0, 1, 2, 3]:  # Background, Vigne, Verger, Vide
        count = class_counts.get(cls, 1)
        frequency = count / total_pixels
        weight = 1.0 / (frequency + 1e-6)  # Éviter division par 0
        weights[cls] = weight
        
        class_name = {0: 'Background', 1: 'Vigne', 2: 'Verger', 3: 'Vide'}[cls]
        print(f"   Classe {cls} ({class_name}): {count:,} pixels ({frequency:.1%}) → poids: {weight:.2f}")
    
    # Normaliser les poids
    max_weight = max(weights.values())
    normalized_weights = {cls: w/max_weight for cls, w in weights.items()}
    
    print(f"\n⚖️  POIDS NORMALISÉS :")
    for cls, weight in normalized_weights.items():
        class_name = {0: 'Background', 1: 'Vigne', 2: 'Verger', 3: 'Vide'}[cls]
        print(f"   Classe {cls} ({class_name}): {weight:.3f}")
    
    return normalized_weights

def create_weighted_sampler(dataset, class_weights=None):
    """
    Crée un sampler pondéré pour équilibrer les classes
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    
    if class_weights is None:
        class_weights = analyze_dataset_classes(dataset)
    
    print("🎯 Création du sampler pondéré...")
    
    # Calculer le poids de chaque échantillon
    sample_weights = []
    
    for i in range(len(dataset)):
        try:
            (data, dates), target = dataset[i]
            
            # Déterminer la classe dominante dans l'échantillon
            semantic_labels = target[:, :, 6].flatten()
            unique, counts = torch.unique(semantic_labels, return_counts=True)
            
            # Classe avec le plus de pixels (hors background si possible)
            dominant_class = 0
            max_count = 0
            
            for cls, count in zip(unique.tolist(), counts.tolist()):
                if cls != 0 and count > max_count:  # Priorité aux non-background
                    dominant_class = cls
                    max_count = count
                elif cls == 0 and dominant_class == 0:  # Background seulement si rien d'autre
                    dominant_class = cls
                    max_count = count
            
            sample_weights.append(class_weights.get(dominant_class, 1.0))
            
            if (i + 1) % 100 == 0:
                print(f"   Échantillons traités: {i+1}/{len(dataset)}")
                
        except Exception as e:
            print(f"Erreur échantillon {i}: {e}")
            sample_weights.append(1.0)  # Poids par défaut
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    print(f"✅ Sampler créé avec {len(sample_weights)} échantillons")
    return sampler