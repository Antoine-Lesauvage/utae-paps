import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.utae import UTAE
from .paps import PAPs

class VineOrchardUTAEPAPs(nn.Module):
    """UTAE+PAPs spécialisé pour vignes et vergers"""
    
    def __init__(self, pretrained_model_path=None, freeze_components=['encoder'], 
                 vine_orchard_classes=3):
        super().__init__()
        
        # Charger le modèle UTAE+PAPs pré-entraîné complet
        if pretrained_model_path:
            self._load_pretrained_model(pretrained_model_path)
        
        # Geler les composants spécifiés
        self._freeze_components(freeze_components)
        
        # Ajouter les têtes spécialisées
        self.vine_orchard_classes = vine_orchard_classes
        self._build_specialized_heads()
    
    def _load_pretrained_model(self, path):
        """Charge le modèle UTAE+PAPs pré-entraîné complet"""
        print(f"Chargement du modèle pré-entraîné : {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # Si c'est un checkpoint d'entraînement, extraire le state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Charger dans self (sera utilisé par les composants hérités)
        self.pretrained_state_dict = state_dict
        self.model_config = checkpoint.get('model_config', {})
        
        print("✓ Modèle pré-entraîné chargé")
    
    def _freeze_components(self, components_to_freeze):
        """Gèle les composants spécifiés"""
        self.frozen_components = components_to_freeze
        print(f"Composants à geler : {components_to_freeze}")
    
    def _build_specialized_heads(self):
        """Construit les têtes spécialisées vignes/vergers"""
        # Dimensions typiques du décodeur UTAE (à ajuster selon votre modèle)
        decoder_dim = 128  # Vous pouvez l'ajuster selon votre config
        
        # Tête spécialisée vignes (motifs linéaires en rangées)
        self.vine_head = nn.Sequential(
            nn.Conv2d(decoder_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Convolution directionnelle pour capturer les rangées de vignes
            nn.Conv2d(256, 128, (1, 7), padding=(0, 3)),  # Horizontal
            nn.Conv2d(128, 128, (7, 1), padding=(3, 0)),  # Vertical
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 2, 1)  # Vigne/Non-vigne
        )
        
        # Tête spécialisée vergers (motifs circulaires des couronnes)
        self.orchard_head = nn.Sequential(
            nn.Conv2d(decoder_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Convolutions avec différents kernels pour capturer les couronnes
            nn.Conv2d(256, 128, 5, padding=2),  # Kernel plus large
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 2, 1)  # Verger/Non-verger
        )
        
        # Tête de fusion pour prédiction finale spécialisée
        self.vine_orchard_fusion = nn.Sequential(
            nn.Conv2d(decoder_dim + 4, 256, 3, padding=1),  # +4 pour vine_head et orchard_head
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, self.vine_orchard_classes, 1)  # Background/Vigne/Verger
        )
        
        print("✓ Têtes spécialisées vignes/vergers créées")
    
    def load_pretrained_weights(self):
        """Charge les poids pré-entraînés et applique le gel"""
        if hasattr(self, 'pretrained_state_dict'):
            # Charger tous les poids compatibles
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in self.pretrained_state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            # Appliquer le gel
            frozen_count = 0
            total_count = 0
            
            for name, param in self.named_parameters():
                total_count += 1
                should_freeze = any(component in name.lower() for component in self.frozen_components)
                
                if should_freeze:
                    param.requires_grad = False
                    frozen_count += 1
                    print(f"❄️  Gelé: {name}")
                else:
                    param.requires_grad = True
                    print(f"🔥 Entraînable: {name}")
            
            print(f"\n✓ {len(pretrained_dict)} poids chargés")
            print(f"✓ {frozen_count}/{total_count} paramètres gelés")
    
    def forward(self, x, batch_positions=None):
        """Forward pass avec prédictions spécialisées"""
        # Cette méthode doit être adaptée selon la structure exacte de votre UTAE+PAPs
        # Vous devrez probablement hériter de votre classe principale et override cette méthode
        
        # Exemple générique - À ADAPTER selon votre modèle :
        # 1. Forward UTAE pour features temporelles
        # utae_features = self.utae_forward(x)
        
        # 2. Forward PAPs si utilisé
        # paps_features = self.paps_forward(utae_features, batch_positions)
        
        # 3. Utiliser les features pour prédictions spécialisées
        # vine_pred = self.vine_head(features)
        # orchard_pred = self.orchard_head(features)
        # fusion_input = torch.cat([features, vine_pred, orchard_pred], dim=1)
        # final_pred = self.vine_orchard_fusion(fusion_input)
        
        # return {'vine_orchard': final_pred, 'vine_specific': vine_pred, 'orchard_specific': orchard_pred}
        
        raise NotImplementedError("À adapter selon votre architecture UTAE+PAPs exacte")