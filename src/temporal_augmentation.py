"""
Temporal augmentation techniques for PASTIS dataset with vine/orchard specialization
"""
import torch
import numpy as np
import random
from typing import Tuple, Dict, Union
from .dataset import PASTIS_Dataset


class TemporalAugmentation:
    def __init__(self):
        pass
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return data, dates


class TemporalShift(TemporalAugmentation):
    def __init__(self, max_shift: int = 3, p: float = 0.5):
        super().__init__()
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return data, dates
        
        shift = random.randint(-self.max_shift, self.max_shift)
        
        if shift != 0:
            aug_data = torch.roll(data, shift, dims=0)
            aug_dates = torch.roll(dates, shift, dims=0) if dates is not None else dates
        else:
            aug_data = data.clone()
            aug_dates = dates.clone() if dates is not None else dates
        
        return aug_data, aug_dates


class TemporalDropout(TemporalAugmentation):
    def __init__(self, dropout_rate: float = 0.1, p: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.p = p
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return data, dates
        
        T = data.shape[0]
        n_dropout = max(1, int(T * self.dropout_rate))
        if T <= 2:
            return data, dates
        
        dropout_indices = random.sample(range(1, T-1), min(n_dropout, T-2))
        aug_data = data.clone()
        
        for idx in dropout_indices:
            aug_data[idx] = (data[idx - 1] + data[idx + 1]) / 2
        
        return aug_data, dates


class TemporalNoise(TemporalAugmentation):
    def __init__(self, noise_std: float = 0.01, p: float = 0.4):
        super().__init__()
        self.noise_std = noise_std
        self.p = p
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return data, dates
        
        noise = torch.randn_like(data) * self.noise_std
        aug_data = data + noise
        return aug_data, dates


class TemporalSubsample(TemporalAugmentation):
    def __init__(self, subsample_rates: list = [2, 3], p: float = 0.3):
        super().__init__()
        self.subsample_rates = subsample_rates
        self.p = p
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return data, dates
        
        T, C, H, W = data.shape
        rate = random.choice(self.subsample_rates)
        
        # Sous-échantillonnage
        subsampled_indices = torch.arange(0, T, rate)
        if len(subsampled_indices) < 2:
            return data, dates
        
        subsampled_data = data[subsampled_indices]
        
        # Interpolation pour revenir à T timesteps
        aug_data = torch.zeros_like(data)
        
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    pixel_series = subsampled_data[:, c, h, w].unsqueeze(0).unsqueeze(0)
                    interpolated = torch.nn.functional.interpolate(
                        pixel_series, size=T, mode='linear', align_corners=True
                    )
                    aug_data[:, c, h, w] = interpolated.squeeze()
        
        return aug_data, dates


class VineOrchardAugmentationPipeline:
    """Pipeline spécialisé pour vignes/vergers avec classes déséquilibrées"""
    
    def __init__(self):
        self.augmentations = [
            TemporalShift(max_shift=2, p=0.6),
            TemporalDropout(dropout_rate=0.03, p=0.4),  
            TemporalNoise(noise_std=0.005, p=0.5),
            TemporalSubsample(subsample_rates=[2], p=0.2),
        ]
    
    def __call__(self, data: torch.Tensor, dates: torch.Tensor, 
                 n_augmentations: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        
        selected_augs = random.sample(
            self.augmentations, 
            min(n_augmentations, len(self.augmentations))
        )
        
        aug_data, aug_dates = data.clone(), dates.clone() if dates is not None else dates
        
        for aug in selected_augs:
            aug_data, aug_dates = aug(aug_data, aug_dates)
        
        return aug_data, aug_dates


class PASTIS_Dataset_WithAugmentation(PASTIS_Dataset):
    def __init__(self, 
                 temporal_aug_pipeline=None,
                 augmentation_factors=None,
                 target_classes=[1, 2, 3],  # Vigne, Verger, Vide
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.temporal_aug = temporal_aug_pipeline or VineOrchardAugmentationPipeline()
        
        # Facteurs d'augmentation par classe
        self.augmentation_factors = augmentation_factors or {
            0: 1,    # Background: pas d'augmentation
            1: 6,    # Vigne: x6
            2: 12,   # Verger: x12 (classe la plus rare)
            3: 3     # Vide: x3
        }
        
        self.target_classes = set(target_classes)
        self.original_len = super().__len__()
        
        # Analyser et catégoriser les échantillons
        self._analyze_samples_by_class()
        
        # Calculer la nouvelle taille
        self._calculate_augmented_size()
    
    def _analyze_samples_by_class(self):
        """Catégorise les échantillons selon leurs classes dominantes"""
        print("🔍 Analyse des échantillons par classe dominante...")
        
        self.samples_by_class = {cls: [] for cls in [0, 1, 2, 3]}
        
        # Analyser un sous-ensemble
        sample_size = min(200, self.original_len)
        indices = random.sample(range(self.original_len), sample_size)
        
        for idx in indices:
            try:
                (data, dates), target = super().__getitem__(idx)
                
                if self.target == "instance":
                    semantic_labels = target[:, :, 6]  # Labels sémantiques
                else:
                    semantic_labels = target
                
                # Identifier la classe dominante (priorité aux classes rares)
                unique_classes, counts = torch.unique(semantic_labels.long(), return_counts=True)
                
                # Priorité maximale aux classes rares
                if 2 in unique_classes:  # Verger
                    dominant_class = 2
                elif 1 in unique_classes:  # Vigne
                    dominant_class = 1
                elif 3 in unique_classes:  # Vide
                    dominant_class = 3
                else:
                    dominant_class = 0  # Background
                
                self.samples_by_class[dominant_class].append(idx)
                
            except Exception as e:
                continue
        
        # Affichage
        class_names = {0: 'Background', 1: 'Vigne', 2: 'Verger', 3: 'Vide'}
        print("📊 RÉPARTITION DES ÉCHANTILLONS PAR CLASSE DOMINANTE:")
        for cls, samples in self.samples_by_class.items():
            factor = self.augmentation_factors.get(cls, 1)
            print(f"   Classe {cls} ({class_names[cls]}): {len(samples)} échantillons → x{factor}")
    
    def _calculate_augmented_size(self):
        """Calcule la taille du dataset avec augmentations ciblées"""
        total_augmented = self.original_len  # Échantillons originaux
        
        for cls, samples in self.samples_by_class.items():
            factor = self.augmentation_factors.get(cls, 1)
            if factor > 1:
                total_augmented += len(samples) * (factor - 1)
        
        self.total_len = total_augmented
        print(f"📈 Dataset original: {self.original_len}")
        print(f"📈 Dataset augmenté: {self.total_len}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        if item < self.original_len:
            return super().__getitem__(item)
        else:
            # Échantillon augmenté - sélection intelligente par classe
            class_probs = [0.1, 0.3, 0.5, 0.1]  # [Background, Vigne, Verger, Vide]
            selected_class = np.random.choice([0, 1, 2, 3], p=class_probs)
            
            # Sélectionner un échantillon de cette classe
            if len(self.samples_by_class[selected_class]) > 0:
                original_idx = random.choice(self.samples_by_class[selected_class])
            else:
                original_idx = random.randint(0, self.original_len - 1)
            
            # Récupérer et augmenter
            (data, dates), target = super().__getitem__(original_idx)
            
            # Augmentations plus agressives pour classes rares
            if selected_class == 2:  # Verger
                n_augs = 3
            elif selected_class == 1:  # Vigne
                n_augs = 2
            else:
                n_augs = 1
            
            # Appliquer les augmentations temporelles
            if isinstance(data, dict):
                aug_data = {}
                aug_dates = {}
                for sat in data.keys():
                    aug_data[sat], aug_dates[sat] = self.temporal_aug(
                        data[sat], dates[sat], n_augmentations=n_augs
                    )
            else:
                aug_data, aug_dates = self.temporal_aug(
                    data, dates, n_augmentations=n_augs
                )
            
            return (aug_data, aug_dates), target


# Fonction utilitaire
def create_vine_orchard_augmentation():
    """Crée un pipeline optimisé pour vignes/vergers"""
    return VineOrchardAugmentationPipeline()