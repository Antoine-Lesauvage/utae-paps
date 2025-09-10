import collections.abc
import re

import torch
from torch.nn import functional as F
from torch.utils import data

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# TRANSFERT LEARNING VIGNES/VERGERS - Mapping des classes
# =============================================================================

# Mapping original (20 classes) -> nouveau (3 classes)
VINE_ORCHARD_CLASS_MAPPING = {
    0: 0,   # Arrière-plan -> Background
    1: 0,   # Prairie -> Background
    2: 0,   # Blé tendre d'hiver -> Background
    3: 0,   # Maïs -> Background
    4: 0,   # Orge d'hiver -> Background
    5: 0,   # Colza d'hiver -> Background
    6: 0,   # Orge de printemps -> Background
    7: 0,   # Tournesol -> Background
    8: 2,   # Vigne -> Vigne ✓
    9: 0,   # Betterave -> Background
    10: 0,  # Triticale d'hiver -> Background
    11: 0,  # Blé dur d'hiver -> Background
    12: 0,  # Fruits, légumes, fleurs -> Background
    13: 0,  # Pommes de terre -> Background
    14: 0,  # Fourrage légumineux -> Background
    15: 0,  # Soja -> Background
    16: 1,  # Verger -> Verger ✓
    17: 0,  # Céréales mélangées -> Background
    18: 0,  # Sorgho -> Background
    19: 0,  # Label vide -> Background
}

# Noms des nouvelles classes
VINE_ORCHARD_CLASS_NAMES = {
    0: 'Background',
    1: 'Verger', 
    2: 'Vigne'
}

# Constantes
NUM_VINE_ORCHARD_CLASSES = 3
VINE_ORCHARD_VOID_LABEL = 0  # Background sera le void_label
VINE_ORCHARD_BACKGROUND_LABEL = 0

def adapt_model_for_vine_orchard(model, pretrained_weights_path=None):
    """
    Adapte un modèle UTAE+PAPS pré-entraîné (20 classes) pour la spécialisation vignes/vergers (3 classes).
    
    Args:
        model: Modèle UTAE+PAPS avec 20 classes
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
        if 'encoder.out_conv.conv.4.weight' in pretrained_state_dict:
            layers_to_adapt.append('encoder.out_conv.conv.4')
            
        # Dans PaPs : class_mlp (dernière couche de classification)
        if 'class_mlp.3.weight' in pretrained_state_dict:
            layers_to_adapt.append('class_mlp.3')
            
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
        
        # 5. Initialiser les nouvelles couches finales
        print("🆕 Initialisation des nouvelles couches finales...")
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in ['out_conv', 'class_mlp']):
                if hasattr(module, 'weight') and module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)
                    print(f"🎲 Initialisé : {name}")
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
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