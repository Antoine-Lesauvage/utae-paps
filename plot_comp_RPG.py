import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
import boto3
import s3fs
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import io
import os
os.environ["AWS_ACCESS_KEY_ID"] = 'BS0G2KO2FTSK62F36PRV'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'lXtVTVZr29H+aP+NNtWxnJv9g1zVkvwD0k5kkJ1M'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJCUzBHMktPMkZUU0s2MkYzNlBSViIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU2MzY2MDAwLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU2OTcyNTkzLCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NjM2Nzc5MiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDo5OGUwY2U5Mi1jZmRkLTA0ZGEtM2FhNi0yMDQxZTgyNzYzZDMiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJlNDYxYTBkYS0wNzkxLTRmYTctOThmMi0wNGFmZjZhMGY2NDMiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.p1E_oPr45DsLUk8BNnApG8__huhL3UtKHdoxyv3Nle0p4jL6atUm964C3RS9OAay7BO9DPWtrtV6WPuxmddxHA'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
class VisualisateurSegmentationRPG_Onyxia:
    def __init__(self, bucket_name, sous_dossier, chemin_rpg, chemin_geojson_tuiles):
        """
        bucket_name: nom du bucket S3 (ex: "antoinelesauvage")
        sous_dossier: chemin dans le bucket (ex: "vergers-france/preds_128_10")
        chemin_rpg: chemin vers le fichier RPG 
        chemin_geojson_tuiles: chemin vers le fichier GeoJSON avec les coordonnées des tuiles
        """
        # Configuration S3 pour Onyxia
        self.fs = s3fs.S3FileSystem(
            endpoint_url='https://minio.lab.sspcloud.fr',
            key=None,
            secret=None
        )
        self.bucket_name = bucket_name
        self.sous_dossier = sous_dossier
        
        # Charger le RPG
        self.rpg = gpd.read_file(chemin_rpg) if isinstance(chemin_rpg, str) else chemin_rpg
        
        # Charger les coordonnées des tuiles depuis le GeoJSON
        self.tuiles_gdf = gpd.read_file(chemin_geojson_tuiles)
        print(f"Chargé {len(self.tuiles_gdf)} tuiles depuis {chemin_geojson_tuiles}")
        print(f"CRS des tuiles: {self.tuiles_gdf.crs}")
        
        # Vérifier que les CRS correspondent, sinon reprojeter
        if self.rpg.crs != self.tuiles_gdf.crs:
            print(f"Reprojection du RPG de {self.rpg.crs} vers {self.tuiles_gdf.crs}")
            self.rpg = self.rpg.to_crs(self.tuiles_gdf.crs)
        
        # Définir les classes sémantiques en français
        self.classes_semantiques = {
            0: 'Arrière-plan',
            1: 'Prairie',
            2: 'Blé tendre d\'hiver',
            3: 'Maïs',
            4: 'Orge d\'hiver',
            5: 'Colza d\'hiver',
            6: 'Orge de printemps',
            7: 'Tournesol',
            8: 'Vigne',
            9: 'Betterave',
            10: 'Triticale d\'hiver',
            11: 'Blé dur d\'hiver',
            12: 'Fruits, légumes, fleurs',
            13: 'Pommes de terre',
            14: 'Fourrage légumineux',
            15: 'Soja',
            16: 'Verger',
            17: 'Céréales mélangées',
            18: 'Sorgho',
            19: 'Label vide'
        }
        
        # Version courte pour l'affichage
        self.classes_semantiques_courtes = {
            0: 'Arrière-plan',
            1: 'Prairie',
            2: 'Blé tendre',
            3: 'Maïs',
            4: 'Orge hiver',
            5: 'Colza',
            6: 'Orge print.',
            7: 'Tournesol',
            8: 'Vigne',
            9: 'Betterave',
            10: 'Triticale',
            11: 'Blé dur',
            12: 'Fruits/légumes',
            13: 'Pommes de terre',
            14: 'Fourrage lég.',
            15: 'Soja',
            16: 'Verger',
            17: 'Céréales mix',
            18: 'Sorgho',
            19: 'Label vide'
        }
    
    def lister_tuiles_disponibles(self, limite=None):
        """Liste toutes les tuiles .npy disponibles dans le bucket et les associe aux coordonnées"""
        chemin_complet = f"{self.bucket_name}/{self.sous_dossier}"
        print(f"Recherche des fichiers dans {chemin_complet}...")
        
        try:
            files = self.fs.glob(f"{chemin_complet}/*_best_fold_*.npy")
            print(f"Fichiers .npy trouvés: {len(files)}")
            
            if limite:
                files = files[:limite]
                print(f"Limité aux {limite} premiers fichiers")
            
        except Exception as e:
            print(f"Erreur lors de la recherche de fichiers: {e}")
            files = []
        
        tuiles = []
        
        for i, file in enumerate(files):
            if i % 100 == 0:  # Progress indicator
                print(f"Traitement: {i}/{len(files)}")
                
            nom_fichier = file.split('/')[-1]
            
            # Extraire le numéro de tuile du nom de fichier
            parts = nom_fichier.split('_best_fold_')
            if len(parts) == 2:
                tuile_id_from_filename = parts[0]
                fold_num = parts[1].replace('.npy', '')
                
                # Optionnel: vérifier le tile_id dans le fichier lui-même
                try:
                    data = self.charger_tuile_depuis_s3(file)
                    if 'tile_id' in data:
                        tuile_id_from_data = str(data['tile_id'])
                        # Utiliser l'ID du fichier de données si disponible
                        tuile_id = tuile_id_from_data
                    else:
                        tuile_id = tuile_id_from_filename
                except:
                    # En cas d'erreur, utiliser l'ID du nom de fichier
                    tuile_id = tuile_id_from_filename
                
                # Chercher les coordonnées correspondantes dans le GeoJSON
                tuile_coords = self.obtenir_coords_tuile(tuile_id)
                
                if tuile_coords is not None:
                    tuiles.append({
                        'tuile_id': tuile_id,
                        'fold': fold_num,
                        'chemin_s3': file,
                        'nom_fichier': nom_fichier,
                        'geometry': tuile_coords['geometry'],
                        'xmin': tuile_coords['xmin'],
                        'ymin': tuile_coords['ymin'],
                        'xmax': tuile_coords['xmax'],
                        'ymax': tuile_coords['ymax']
                    })
        
        print(f"Total de tuiles avec coordonnées: {len(tuiles)}")
        return pd.DataFrame(tuiles)
    
    def obtenir_coords_tuile(self, tuile_id):
        """Récupère les coordonnées d'une tuile depuis le GeoJSON"""
        # Conversion de l'ID en entier
        try:
            tuile_id_int = int(tuile_id)
        except ValueError:
            print(f"Impossible de convertir {tuile_id} en entier")
            return None
        
        # Chercher par l'ID dans les propriétés
        mask = self.tuiles_gdf['id'] == tuile_id_int
        
        if mask.any():
            row = self.tuiles_gdf[mask].iloc[0]
            geom = row.geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            
            return {
                'geometry': geom,
                'xmin': bounds[0],
                'ymin': bounds[1],
                'xmax': bounds[2],
                'ymax': bounds[3]
            }
        else:
            return None
    
    def charger_tuile_depuis_s3(self, chemin_s3):
        """Charge un fichier .npy depuis le S3"""
        with self.fs.open(chemin_s3, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            if hasattr(data, 'item') and isinstance(data.item(), dict):
                return data.item()
            else:
                return data
    
    def visualiser_tuile_complete(self, tuile_id, fold='5', figsize=(28, 14)):
        """Visualise une tuile complète avec toutes les informations"""
        
        # Construire le chemin S3
        chemin_s3 = f"{self.bucket_name}/{self.sous_dossier}/{tuile_id}_best_fold_{fold}.npy"
        
        # Charger les données
        print(f"Chargement de {chemin_s3}...")
        try:
            data = self.charger_tuile_depuis_s3(chemin_s3)
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None, None
        
        # Obtenir les coordonnées
        coords_info = self.obtenir_coords_tuile(tuile_id)
        if coords_info is None:
            print(f"Impossible de trouver les coordonnées pour la tuile {tuile_id}")
            return None, None
        
        # Filtrer le RPG
        rpg_filtre = self.filtrer_rpg_par_tuile(coords_info['geometry'])
        print(f"RPG filtré: {len(rpg_filtre)} parcelles trouvées")
        
        # Créer la figure avec plusieurs sous-graphiques
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Segmentation panoptique (instances)
        self._afficher_pano_instance(axes[0], data['pano_instance'], 
                                   f"Instances panoptiques - Tuile {tuile_id}")
        
        # 2. Segmentation sémantique
        self._afficher_pano_semantic(axes[1], data['pano_semantic'], 
                                   f"Segmentation sémantique - Tuile {tuile_id}")
        
        # 3. RPG seul
        self._afficher_rpg(axes[2], rpg_filtre, coords_info, 
                          f"RPG - Tuile {tuile_id}")
        
        # 4. Superposition instances + RPG
        self._afficher_superposition(axes[3], data['pano_instance'], rpg_filtre, 
                                   coords_info, f"Instances + RPG - Tuile {tuile_id}")
        
        # 5. Superposition sémantique + RPG
        self._afficher_superposition_semantic(axes[4], data['pano_semantic'], rpg_filtre, 
                                            coords_info, f"Sémantique + RPG - Tuile {tuile_id}")
        
        # 6. Heatmap de confiance
        if 'confidence' in data:
            self._afficher_confidence(axes[5], data['confidence'], 
                                    f"Confiance - Tuile {tuile_id}")
        else:
            axes[5].axis('off')
            axes[5].text(0.5, 0.5, 'Pas de données\nde confiance', 
                        ha='center', va='center', transform=axes[5].transAxes)
        #if len(rpg_filtre) > 0:
            #diagnostic = self.diagnostic_complet_differences(
            #tuile_id, data, coords_info, rpg_filtre
            #)
        
            #if not diagnostic['georef_ok']:
                #print("🚨 Problème de géoréférencement détecté !")
            #elif not diagnostic['resolution_ok']:
                #print("🚨 Problème de déformation détecté !")
            #else:
                #print("✅ Transformations géométriques OK - Différences probablement dues au modèle")
        plt.suptitle(f"Analyse complète - Tuile {tuile_id} (Fold {fold})", fontsize=16)
        plt.tight_layout()
    # Ajuster l'espacement pour faire de la place aux légendes
        plt.tight_layout()
    # Plus d'espace à droite pour les légendes
        plt.subplots_adjust(right=0.75, wspace=0.4, hspace=0.4, top=0.93)
        return fig, data

    def ajouter_legende_globale_semantique(self, fig):
        """Ajoute une légende globale pour toutes les classes sémantiques"""
    
    # Créer une légende avec toutes les classes
        n_classes = len(self.classes_semantiques)
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
        from matplotlib.patches import Patch
        legend_elements = []
    
        for classe_id, nom_classe in self.classes_semantiques_courtes.items():
            if classe_id < len(colors):
                legend_elements.append(Patch(facecolor=colors[classe_id], label=nom_classe))
    
    # Ajouter la légende à la figure
        fig.legend(handles=legend_elements,
              bbox_to_anchor=(0.98, 0.5),
              loc='center right',
              fontsize=8,
              title='Classes sémantiques',
              title_fontsize=10)
    # ... (garder toutes les autres méthodes d'affichage identiques)
    def filtrer_rpg_par_tuile(self, geometry_tuile):
        """Filtre les polygones RPG qui intersectent avec la tuile"""
        tuile_gdf = gpd.GeoDataFrame([1], geometry=[geometry_tuile], crs=self.tuiles_gdf.crs)
        return gpd.overlay(self.rpg, tuile_gdf, how='intersection')

    def _afficher_pano_instance(self, ax, pano_instance, titre):
        """Affiche la segmentation panoptique d'instances avec légende"""
        if len(pano_instance.shape) == 3 and pano_instance.shape[0] == 1:
            pano_instance = pano_instance[0]
    
        unique_instances = np.unique(pano_instance)
        unique_instances = unique_instances[unique_instances != 0]  # Exclure le background
        n_instances = len(unique_instances)
    
    # Choisir la colormap
        if n_instances > 20:
            cmap = plt.cm.viridis
        else:
            cmap = plt.cm.tab20
    
        im = ax.imshow(pano_instance, cmap=cmap, vmin=0, vmax=max(unique_instances) if len(unique_instances) > 0 else 1)
        ax.set_title(titre)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
    
    # ===== LÉGENDE PANOPTIQUE =====
        if n_instances > 0 and n_instances <= 15:  # Afficher légende seulement si pas trop d'instances
            from matplotlib.patches import Patch
            legend_elements = []
        
        # Ajouter le background
            if 0 in np.unique(pano_instance):
                legend_elements.append(Patch(facecolor=cmap(0), label='Background'))
        
        # Ajouter les instances
            for i, instance_id in enumerate(sorted(unique_instances)[:15]):  # Limiter à 15
            # Calculer la taille de l'instance
                taille_instance = np.sum(pano_instance == instance_id)
                pourcentage = (taille_instance / pano_instance.size) * 100
            
            # Couleur selon la colormap utilisée
                if n_instances <= 20:
                    couleur = cmap(i / 19)  # tab20 normalisé
                else:
                    couleur = cmap(instance_id / max(unique_instances))
            
                label = f'Inst. {instance_id} ({pourcentage:.1f}%)'
                legend_elements.append(Patch(facecolor=couleur, label=label))
        
        # Ajouter la légende
            ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left',
                 fontsize=7,
                 title=f'Instances ({n_instances})')
    
    # Texte informatif si trop d'instances pour la légende
        elif n_instances > 15:
            ax.text(0.02, 0.98, f'{n_instances} instances\n(trop pour légende)', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.98, f'{n_instances} instances', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    def _afficher_pano_semantic(self, ax, pano_semantic, titre):
        """Affiche la segmentation sémantique avec légende"""
    # Traitement de la segmentation sémantique
        if len(pano_semantic.shape) == 4:
            semantic_map = np.argmax(pano_semantic[0], axis=0)
        elif len(pano_semantic.shape) == 3:
            semantic_map = np.argmax(pano_semantic, axis=0)
        else:
            semantic_map = pano_semantic
    
    # Créer une colormap pour les classes sémantiques
        n_classes = len(self.classes_semantiques)
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        cmap = mcolors.ListedColormap(colors)
    
        im = ax.imshow(semantic_map, cmap=cmap, vmin=0, vmax=n_classes-1)
        ax.set_title(titre)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
    
    # ===== NOUVELLE PARTIE : LÉGENDE =====
    # Créer la légende uniquement pour les classes présentes
        classes_presentes = np.unique(semantic_map)
    
    # Créer les patches pour la légende
        from matplotlib.patches import Patch
        legend_elements = []
    
        for classe_id in sorted(classes_presentes):
            if classe_id < len(colors):
                nom_classe = self.classes_semantiques_courtes.get(classe_id, f'Classe_{classe_id}')
            # Calculer le pourcentage de pixels
                pourcentage = (np.sum(semantic_map == classe_id) / semantic_map.size) * 100
                label = f'{nom_classe} ({pourcentage:.1f}%)'
                legend_elements.append(Patch(facecolor=colors[classe_id], label=label))
    
    # Ajouter la légende à côté du graphique
        ax.legend(handles=legend_elements, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              fontsize=8,
              frameon=True,
              fancybox=True,
              shadow=True)

    def _afficher_rpg(self, ax, rpg_filtre, coords_info, titre):
        """Affiche les polygones RPG avec légende aux couleurs cohérentes"""
        if not rpg_filtre.empty:
        # Vérifier si on a une colonne de culture
            if 'CODE_CULTU' in rpg_filtre.columns:
            # Obtenir les cultures uniques et leur assigner des couleurs fixes
                cultures_presentes = sorted(rpg_filtre['CODE_CULTU'].unique())
                n_cultures = len(cultures_presentes)
            
            # Créer un mapping culture -> couleur fixe
                cmap = plt.cm.tab10
                couleurs_cultures = {}
                for i, culture in enumerate(cultures_presentes):
                    couleurs_cultures[culture] = cmap(i / max(9, n_cultures-1))  # tab10 a 10 couleurs
            
            # Créer une colonne de couleurs pour chaque parcelle
                rpg_filtre = rpg_filtre.copy()
                rpg_filtre['couleur'] = rpg_filtre['CODE_CULTU'].map(couleurs_cultures)
            
            # Afficher chaque culture avec sa couleur
                for culture in cultures_presentes:
                    mask = rpg_filtre['CODE_CULTU'] == culture
                    if mask.any():
                        rpg_filtre[mask].plot(ax=ax, color=couleurs_cultures[culture], 
                                        alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # ===== LÉGENDE RPG CORRIGÉE =====
                from matplotlib.patches import Patch
                legend_elements = []
            
                for culture in cultures_presentes:
                    nom_culture = self._obtenir_nom_culture(culture)
                    nb_parcelles = np.sum(rpg_filtre['CODE_CULTU'] == culture)
                    label = f'{nom_culture} ({nb_parcelles})'
                    legend_elements.append(Patch(facecolor=couleurs_cultures[culture], 
                                           edgecolor='black',
                                           label=label))
            
            # Ajouter la légende
                ax.legend(handles=legend_elements,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     fontsize=8,
                     title='Cultures RPG')
                     
            else:
            # Si pas de colonne culture, afficher en une seule couleur
                rpg_filtre.plot(ax=ax, alpha=0.7, edgecolor='red', 
                          facecolor='lightblue')
            
            # Légende simple
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='lightblue', edgecolor='red', 
                                   label=f'Parcelles RPG ({len(rpg_filtre)})')]
                ax.legend(handles=legend_elements,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     fontsize=8)
    
    # Ajouter la bbox de la tuile
        tuile_gdf = gpd.GeoDataFrame([1], geometry=[coords_info['geometry']], 
                               crs=self.tuiles_gdf.crs)
        tuile_gdf.boundary.plot(ax=ax, color='blue', linewidth=2)
    
        ax.set_title(titre)
        ax.set_aspect('equal')
    
    # Texte informatif
        ax.text(0.02, 0.02, f'{len(rpg_filtre)} parcelles', 
            transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    def _obtenir_nom_culture(self, code_culture):
        """Convertit un code culture RPG en nom lisible (version étendue)"""
    # Dictionnaire de mapping étendu des codes RPG
        mapping_cultures_rpg = {
        # Céréales
        'BTH': 'Blé tendre hiver',
        'BTD': 'Blé dur',
        'ORH': 'Orge hiver',
        'ORP': 'Orge printemps',
        'MID': 'Maïs grain',
        'AVH': 'Avoine hiver',
        'AVP': 'Avoine printemps',
        'TTH': 'Triticale hiver',
        'RIZ': 'Riz',
        'SOR': 'Sorgho',
        
        # Oléagineux
        'COL': 'Colza',
        'TRN': 'Tournesol',
        'SOJ': 'Soja',
        'LIN': 'Lin oléagineux',
        
        # Protéagineux
        'POI': 'Pois protéagineux',
        'FEV': 'Féverole',
        'LUP': 'Lupin',
        
        # Prairies
        'PPH': 'Prairie permanente',
        'PTR': 'Prairie temporaire',
        'RGA': 'Ray-grass',
        'LUZ': 'Luzerne',
        'TRE': 'Trèfle',
        
        # Cultures spécialisées
        'VER': 'Vergers',
        'VIG': 'Vignes',
        'BET': 'Betterave sucrière',
        'PDT': 'Pommes de terre',
        'LPC': 'Légumes plein champ',
        'FLO': 'Fleurs',
        
        # Autres
        'JAC': 'Jachère',
        'FOR': 'Forêt',
        'DIV': 'Divers',
        }
    
        return mapping_cultures_rpg.get(str(code_culture).upper(), f'Culture_{code_culture}')
 
    def _afficher_superposition(self, ax, pano_instance, rpg_filtre, coords_info, titre):
        """Superpose instances panoptiques et RPG avec légendes cohérentes"""
        if len(pano_instance.shape) == 3 and pano_instance.shape[0] == 1:
            pano_instance = pano_instance[0]
    
        unique_instances = np.unique(pano_instance)
        unique_instances = unique_instances[unique_instances != 0]
        n_instances = len(unique_instances)
    
    # Choisir la même colormap que pour l'affichage individuel
        if n_instances > 20:
            cmap = plt.cm.viridis
        else:
            cmap = plt.cm.tab20
    
        ax.imshow(pano_instance, cmap=cmap, alpha=0.6, 
             vmin=0, vmax=max(unique_instances) if len(unique_instances) > 0 else 1)
    
    # Superposer le RPG avec couleurs cohérentes
        if not rpg_filtre.empty:
            rpg_pixel = self._geo_vers_pixel(rpg_filtre, coords_info, pano_instance.shape)
        
        # Si on a des codes culture, utiliser les mêmes couleurs que dans _afficher_rpg
            if 'CODE_CULTU' in rpg_filtre.columns:
                cultures_presentes = sorted(rpg_filtre['CODE_CULTU'].unique())
                cmap_rpg = plt.cm.Set3  # Couleurs différentes pour distinguer du panoptique
                couleurs_cultures = {}
                for i, culture in enumerate(cultures_presentes):
                    couleurs_cultures[culture] = cmap_rpg(i / max(11, len(cultures_presentes)-1))
            
            # Dessiner chaque culture avec sa couleur
                for culture in cultures_presentes:
                    mask = rpg_pixel.index[rpg_filtre['CODE_CULTU'] == culture]
                    if len(mask) > 0:
                        rpg_culture = rpg_pixel.loc[mask]
                        self._dessiner_polygones_pixel(ax, rpg_culture, 
                                                 color=couleurs_cultures[culture], 
                                                 linewidth=2)
            else:
                self._dessiner_polygones_pixel(ax, rpg_pixel, color='red', linewidth=2)
    
        ax.set_title(titre)
        ax.set_xlim(0, pano_instance.shape[1])
        ax.set_ylim(pano_instance.shape[0], 0)
    
    # ===== LÉGENDE COMBINÉE =====
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
    
    # Ajouter quelques instances principales (les plus grandes)
        if n_instances > 0 and n_instances <= 10:
            for i, instance_id in enumerate(sorted(unique_instances)[:5]):  # Top 5
                taille_instance = np.sum(pano_instance == instance_id)
                pourcentage = (taille_instance / pano_instance.size) * 100
            
                if n_instances <= 20:
                    couleur = cmap(i / 19)
                else:
                    couleur = cmap(instance_id / max(unique_instances))
            
                legend_elements.append(Patch(facecolor=couleur, alpha=0.6,
                                       label=f'Inst. {instance_id} ({pourcentage:.1f}%)'))
    
    # Ajouter les éléments RPG
        if not rpg_filtre.empty:
            if 'CODE_CULTU' in rpg_filtre.columns and len(cultures_presentes) <= 5:
                for culture in cultures_presentes:
                    nom_culture = self._obtenir_nom_culture(culture)
                    nb_parcelles = np.sum(rpg_filtre['CODE_CULTU'] == culture)
                    legend_elements.append(Line2D([0], [0], 
                                            color=couleurs_cultures[culture], 
                                            linewidth=3,
                                            label=f'RPG: {nom_culture} ({nb_parcelles})'))
            else:
                legend_elements.append(Line2D([0], [0], color='red', linewidth=3,
                                        label=f'RPG ({len(rpg_filtre)} parcelles)'))
    
    # Ajouter la légende si pas trop d'éléments
        if len(legend_elements) <= 10:
            ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left',
                 fontsize=7,
                 title='Instances + RPG')
    def _afficher_superposition_semantic(self, ax, pano_semantic, rpg_filtre, coords_info, titre):
        """Superpose segmentation sémantique et RPG avec légendes"""
    # Traiter la segmentation sémantique
        if len(pano_semantic.shape) == 4:
            semantic_map = np.argmax(pano_semantic[0], axis=0)
        elif len(pano_semantic.shape) == 3:
            semantic_map = np.argmax(pano_semantic, axis=0)
        else:
            semantic_map = pano_semantic
    
    # Afficher la segmentation sémantique
        n_classes = len(self.classes_semantiques)
        colors_semantic = plt.cm.tab20(np.linspace(0, 1, n_classes))
        cmap = mcolors.ListedColormap(colors_semantic)
    
        ax.imshow(semantic_map, cmap=cmap, alpha=0.6, vmin=0, vmax=n_classes-1)
    
    # Superposer le RPG
        if not rpg_filtre.empty:
            rpg_pixel = self._geo_vers_pixel(rpg_filtre, coords_info, semantic_map.shape)
            self._dessiner_polygones_pixel(ax, rpg_pixel, color='red', linewidth=2)
    
        ax.set_title(titre)
        ax.set_xlim(0, semantic_map.shape[1])
        ax.set_ylim(semantic_map.shape[0], 0)
    
    # ===== LÉGENDE COMBINÉE =====
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
    
    # Ajouter les classes sémantiques présentes
        classes_presentes = np.unique(semantic_map)
        for classe_id in sorted(classes_presentes)[:5]:  # Limiter à 5 pour éviter surcharge
            if classe_id < len(colors_semantic):
                nom_classe = self.classes_semantiques_courtes.get(classe_id, f'Classe_{classe_id}')
                legend_elements.append(Patch(facecolor=colors_semantic[classe_id], 
                                       label=f'Sém: {nom_classe}'))
    
    # Ajouter l'élément RPG
        if not rpg_filtre.empty:
            legend_elements.append(Line2D([0], [0], color='red', linewidth=2, 
                                    label=f'RPG ({len(rpg_filtre)} parcelles)'))
    
    # Ajouter la légende
        ax.legend(handles=legend_elements,
             bbox_to_anchor=(1.05, 1),
             loc='upper left',
             fontsize=7,
             title='Classes')
    def diagnostiquer_georef(self, tuile_id, data, coords_info, rpg_filtre):
        """Diagnostique les problèmes de géoréférencement"""
        
        print(f"\n=== DIAGNOSTIC GÉORÉFÉRENCEMENT - Tuile {tuile_id} ===")
        
        # Vérifications de base
        print(f"Coordonnées tuile (Lambert 93):")
        print(f"  xmin: {coords_info['xmin']:.2f}, ymin: {coords_info['ymin']:.2f}")
        print(f"  xmax: {coords_info['xmax']:.2f}, ymax: {coords_info['ymax']:.2f}")
        print(f"  Largeur: {coords_info['xmax'] - coords_info['xmin']:.2f} m")
        print(f"  Hauteur: {coords_info['ymax'] - coords_info['ymin']:.2f} m")
        
        # Résolution pixel
        pano_instance = data['pano_instance']
        if len(pano_instance.shape) == 3:
            pano_instance = pano_instance[0]
        
        h_pixels, w_pixels = pano_instance.shape
        resolution_x = (coords_info['xmax'] - coords_info['xmin']) / w_pixels
        resolution_y = (coords_info['ymax'] - coords_info['ymin']) / h_pixels
        
        print(f"Résolution calculée:")
        print(f"  X: {resolution_x:.2f} m/pixel")
        print(f"  Y: {resolution_y:.2f} m/pixel")
        print(f"  Dimensions image: {h_pixels}x{w_pixels} pixels")
        
        # Vérifier la cohérence
        if abs(resolution_x - resolution_y) > 0.1:
            print("⚠️  ATTENTION: Résolutions X et Y différentes - possible déformation")
        
        overlap_ratio = 0
        if not rpg_filtre.empty:
            # Analyser les parcelles RPG
            rpg_bounds = rpg_filtre.total_bounds
            print(f"Bounds RPG:")
            print(f"  xmin: {rpg_bounds[0]:.2f}, ymin: {rpg_bounds[1]:.2f}")
            print(f"  xmax: {rpg_bounds[2]:.2f}, ymax: {rpg_bounds[3]:.2f}")
            
            # Vérifier le recouvrement
            overlap_x = min(coords_info['xmax'], rpg_bounds[2]) - max(coords_info['xmin'], rpg_bounds[0])
            overlap_y = min(coords_info['ymax'], rpg_bounds[3]) - max(coords_info['ymin'], rpg_bounds[1])
            
            tuile_area = (coords_info['xmax'] - coords_info['xmin']) * (coords_info['ymax'] - coords_info['ymin'])
            overlap_area = max(0, overlap_x) * max(0, overlap_y)
            overlap_ratio = overlap_area / tuile_area
            
            print(f"Recouvrement tuile/RPG: {overlap_ratio:.1%}")
            
            if overlap_ratio < 0.9:
                print("⚠️  ATTENTION: Recouvrement partiel - possible décalage géographique")
        
        return {
            'resolution_x': resolution_x,
            'resolution_y': resolution_y,
            'overlap_ratio': overlap_ratio
        }

    def tester_transformation_pixel_geo(self, coords_info, shape_image):
        """Test la précision de la transformation pixel <-> géo"""
        
        height, width = shape_image
        
        # Points de test en pixels
        points_test_pixel = [
            (0, 0),                    # Coin haut-gauche
            (width-1, 0),              # Coin haut-droite  
            (0, height-1),             # Coin bas-gauche
            (width-1, height-1),       # Coin bas-droite
            (width//2, height//2)      # Centre
        ]
        
        print(f"\n=== TEST TRANSFORMATION PIXEL ↔ GEO ===")
        
        for i, (px, py) in enumerate(points_test_pixel):
            # Pixel -> Géo
            geo_x = coords_info['xmin'] + (px / width) * (coords_info['xmax'] - coords_info['xmin'])
            geo_y = coords_info['ymax'] - (py / height) * (coords_info['ymax'] - coords_info['ymin'])
            
            # Géo -> Pixel (vérification)
            px_back = (geo_x - coords_info['xmin']) / (coords_info['xmax'] - coords_info['xmin']) * width
            py_back = (coords_info['ymax'] - geo_y) / (coords_info['ymax'] - coords_info['ymin']) * height
            
            print(f"Point {i+1}: Pixel({px}, {py}) -> Géo({geo_x:.2f}, {geo_y:.2f}) -> Pixel({px_back:.1f}, {py_back:.1f})")
            
            # Vérifier la précision
            erreur_px = abs(px - px_back)
            erreur_py = abs(py - py_back)
            
            if erreur_px > 0.5 or erreur_py > 0.5:
                print(f"⚠️  Erreur de transformation > 0.5 pixel")

    def _extraire_contours_instance(self, mask):
        """Extrait les contours d'une instance (version simplifiée)"""
        # Version basique pour calculer le périmètre
        try:
            from scipy import ndimage
            # Calculer les bordures
            bordures = mask ^ ndimage.binary_erosion(mask)
            contours = np.where(bordures)
            return list(zip(contours[0], contours[1]))
        except:
            # Fallback simple
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                return list(zip(y_coords, x_coords))
            return []

    def analyser_differences_morphologiques(self, data, rpg_filtre, coords_info):
        """Analyse les différences de forme entre instances et RPG"""
        
        if rpg_filtre.empty:
            print("Pas de parcelles RPG pour comparaison")
            return
        
        pano_instance = data['pano_instance']
        if len(pano_instance.shape) == 3:
            pano_instance = pano_instance[0]
        
        unique_instances = np.unique(pano_instance)
        unique_instances = unique_instances[unique_instances != 0]
        
        print(f"\n=== ANALYSE MORPHOLOGIQUE ===")
        print(f"Instances détectées: {len(unique_instances)}")
        print(f"Parcelles RPG: {len(rpg_filtre)}")
        
        # Statistiques des instances
        instance_stats = []
        for instance_id in unique_instances:
            mask = pano_instance == instance_id
            pixels = np.sum(mask)
            
            # Calculer des propriétés géométriques simples
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                # Boîte englobante
                bbox_width = np.max(x_coords) - np.min(x_coords) + 1
                bbox_height = np.max(y_coords) - np.min(y_coords) + 1
                bbox_area = bbox_width * bbox_height
                
                # Compacité (ratio aire/périmètre²)
                contours = self._extraire_contours_instance(mask)
                perimeter = len(contours) if contours else 0
                compacite = (4 * np.pi * pixels) / (perimeter**2) if perimeter > 0 else 0
                
                # Ratio de remplissage de la bbox
                fill_ratio = pixels / bbox_area if bbox_area > 0 else 0
                
                instance_stats.append({
                    'id': instance_id,
                    'pixels': pixels,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'compacite': compacite,
                    'fill_ratio': fill_ratio
                })
        
        # Statistiques des parcelles RPG
        rpg_pixel = self._geo_vers_pixel(rpg_filtre, coords_info, pano_instance.shape)
        rpg_stats = []
        
        for idx, row in rpg_pixel.iterrows():
            if hasattr(row.geometry, 'exterior'):
                rpg_stats.append({
                    'area_approx': row.geometry.area,
                    'bounds': row.geometry.bounds
                })
        
        # Comparaisons
        if instance_stats and rpg_stats:
            avg_instance_size = np.mean([s['pixels'] for s in instance_stats])
            avg_instance_compacite = np.mean([s['compacite'] for s in instance_stats])
            
            print(f"Taille moyenne des instances: {avg_instance_size:.1f} pixels")
            print(f"Compacité moyenne des instances: {avg_instance_compacite:.3f}")
            
            # Identifier les causes probables de différences
            self._identifier_causes_differences(instance_stats, rpg_stats)

    def _identifier_causes_differences(self, instance_stats, rpg_stats):
        """Identifie les causes probables des différences"""
        
        print(f"\n=== CAUSES PROBABLES DES DIFFÉRENCES ===")
        
        # Analyser la variabilité des formes
        compacites = [s['compacite'] for s in instance_stats]
        fill_ratios = [s['fill_ratio'] for s in instance_stats]
        
        compacite_std = np.std(compacites)
        fill_ratio_std = np.std(fill_ratios)
        
        print(f"Variabilité des formes d'instances:")
        print(f"  Écart-type compacité: {compacite_std:.3f}")
        print(f"  Écart-type fill ratio: {fill_ratio_std:.3f}")
        
        # Hypothèses basées sur les observations
        if compacite_std > 0.3:
            print("🔍 Forte variabilité de compacité -> Possible sur-segmentation ou fusion d'objets")
        
        if np.mean(fill_ratios) < 0.7:
            print("🔍 Faible fill ratio -> Instances possiblement fragmentées")
        
        if len(instance_stats) > len(rpg_stats) * 1.5:
            print("🔍 Beaucoup plus d'instances que de parcelles -> Sur-segmentation probable")
        elif len(instance_stats) < len(rpg_stats) * 0.7:
            print("🔍 Moins d'instances que de parcelles -> Sous-segmentation ou fusion")
        
        # Recommandations
        print(f"\n=== RECOMMANDATIONS ===")
        print("1. Vérifier le géoréférencement avec diagnostiquer_georef()")
        print("2. Examiner visuellement les superpositions")
        print("3. Vérifier les paramètres du modèle de segmentation")
        print("4. Considérer un post-traitement des instances")

    def diagnostic_complet_differences(self, tuile_id, data, coords_info, rpg_filtre):
        """Diagnostic complet des différences instances/RPG"""
        
        print(f"🔍 DIAGNOSTIC COMPLET - Tuile {tuile_id}")
        print("="*60)
        
        # 1. Diagnostic géoréférencement
        georef_stats = self.diagnostiquer_georef(tuile_id, data, coords_info, rpg_filtre)
        
        # 2. Test transformation
        pano_instance = data['pano_instance']
        if len(pano_instance.shape) == 3:
            pano_instance = pano_instance[0]
        self.tester_transformation_pixel_geo(coords_info, pano_instance.shape)
        
        # 3. Analyse morphologique
        self.analyser_differences_morphologiques(data, rpg_filtre, coords_info)
        
        # 4. Recommandations finales
        print(f"\n🎯 CONCLUSIONS PROBABLES:")
        
        if georef_stats['overlap_ratio'] < 0.9:
            print("❌ PROBLÈME GÉOGRAPHIQUE: Décalage ou mauvais géoréférencement")
            print("   → Vérifier les coordonnées des tuiles et le CRS")
        
        if abs(georef_stats['resolution_x'] - georef_stats['resolution_y']) > 0.1:
            print("❌ PROBLÈME DE DÉFORMATION: Résolutions X/Y différentes")
            print("   → Vérifier la transformation géométrique")
        
        print("✅ EFFETS MODÈLE POSSIBLES:")
        print("   → Sur/sous-segmentation (normal pour un modèle)")
        print("   → Lissage des contours (résolution finie)")
        print("   → Confusion entre cultures similaires")
        print("   → Biais temporal (date images vs RPG)")
        
        return {
            'georef_ok': georef_stats['overlap_ratio'] > 0.9,
            'resolution_ok': abs(georef_stats['resolution_x'] - georef_stats['resolution_y']) < 0.1,
            'recommendations': []
        }
    
    def _traiter_tuiles_selectionnees(self, tuiles_selectionnees, dossier_sortie):
        """Sépare le traitement des tuiles pour réutilisabilité"""
    
        # Statistiques des tuiles sélectionnées
        parcelles_stats = [t['nb_parcelles'] for t in tuiles_selectionnees]
        print(f"Statistiques parcelles RPG:")
        print(f"  Moyenne: {np.mean(parcelles_stats):.1f}")
        print(f"  Min: {np.min(parcelles_stats)}")
        print(f"  Max: {np.max(parcelles_stats)}")
    
    # Traitement des tuiles
        resultats = []
        succes = 0
        echecs = 0
        nb_a_traiter = len(tuiles_selectionnees)
    
        print(f"\nDébut du traitement de {nb_a_traiter} tuiles...")
        print("=" * 60)
    
        for idx, tuile_data in enumerate(tuiles_selectionnees):
            tuile_id = tuile_data['tuile_id']
            fold = tuile_data['fold']
        
            print(f"[{idx+1}/{nb_a_traiter}] Traitement tuile {tuile_id}...")
        
            try:
            # Visualiser la tuile
                fig, data = self.visualiser_tuile_complete(
                    tuile_id, fold, figsize=(24, 12)
                )
            
                if fig is not None:
                # Nom du fichier de sortie
                    nom_fichier = f"tuile_{tuile_id}_fold_{fold}_parcelles_{tuile_data['nb_parcelles']}.png"
                    chemin_complet = os.path.join(dossier_sortie, nom_fichier)
                
                # Sauvegarder
                    fig.savefig(chemin_complet, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                    print(f"  ✅ Sauvegardé: {nom_fichier}")
                
                # Fermer la figure pour libérer la mémoire
                    plt.close(fig)
                
                # Collecter les statistiques
                    resultats.append({
                    'tuile_id': tuile_id,
                    'fold': fold,
                    'nb_parcelles': tuile_data['nb_parcelles'],
                    'fichier': nom_fichier,
                    'statut': 'succès'
                })
                
                    succes += 1
                
                else:
                    print(f"  ❌ Échec: figure non générée")
                    echecs += 1
                
            except Exception as e:
                print(f"  ❌ Erreur: {str(e)}")
                echecs += 1
            
                resultats.append({
                'tuile_id': tuile_id,
                'fold': fold,
                'nb_parcelles': tuile_data['nb_parcelles'],
                'fichier': None,
                'statut': f'erreur: {str(e)}'
                })
    
    # Résumé final
        print("\n" + "=" * 60)
        print("RÉSUMÉ DU TRAITEMENT")
        print("=" * 60)
        print(f"Tuiles traitées: {nb_a_traiter}")
        print(f"Succès: {succes}")
        print(f"Échecs: {echecs}")
        print(f"Taux de succès: {succes/nb_a_traiter*100:.1f}%")
        print(f"Images sauvegardées dans: {dossier_sortie}/")
    
    # Sauvegarder un rapport
        self._sauvegarder_rapport_batch(resultats, dossier_sortie, tuiles_selectionnees)
    
        return resultats

    def traitement_batch_rapide(self, nb_tuiles=100, dossier_sortie="comp_RPG", 
                          min_parcelles_rpg=1, seed=42):
        """Version la plus rapide possible"""
    
        import os
        import random
    
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
    
        print("🚀 Mode rapide : échantillonnage intelligent")
    
    # ✅ Stratégie : prendre un échantillon aléatoire de tuiles directement
    # au lieu de toutes les charger
    
    # 1. Obtenir la liste complète des fichiers S3 (rapide)
        chemin_complet = f"{self.bucket_name}/{self.sous_dossier}"
        files = self.fs.glob(f"{chemin_complet}/*_best_fold_*.npy")
    
        print(f"Total fichiers disponibles: {len(files)}")
    
    # 2. Échantillonnage aléatoire des fichiers
        random.seed(seed)
    # Prendre 3x plus de fichiers que nécessaire pour avoir une marge
        nb_echantillon = min(nb_tuiles * 3, len(files))
        files_echantillon = random.sample(files, nb_echantillon)
    
        print(f"Échantillon de fichiers: {nb_echantillon}")
    
    # 3. Traiter l'échantillon jusqu'à avoir assez de tuiles valides
        tuiles_valides = []
    
        for idx, file_path in enumerate(files_echantillon):
            if len(tuiles_valides) >= nb_tuiles:
                break
            
        # Extraire l'ID de la tuile du nom de fichier
            nom_fichier = file_path.split('/')[-1]
            parts = nom_fichier.split('_best_fold_')
            if len(parts) == 2:
                tuile_id = parts[0]
                fold = parts[1].replace('.npy', '')
            
            # Vérifier rapidement s'il y a des parcelles RPG
                coords_info = self.obtenir_coords_tuile(tuile_id)
                if coords_info:
                    rpg_filtre = self.filtrer_rpg_par_tuile(coords_info['geometry'])
                
                    if len(rpg_filtre) >= min_parcelles_rpg:
                        tuiles_valides.append({
                        'tuile_id': tuile_id,
                        'fold': fold,
                        'nb_parcelles': len(rpg_filtre),
                        'coords_info': coords_info
                        })
                    
                        if idx % 10 == 0:
                            print(f"  Trouvées: {len(tuiles_valides)}/{nb_tuiles}")
    
        print(f"✅ {len(tuiles_valides)} tuiles valides trouvées")
    
        if len(tuiles_valides) == 0:
            print("❌ Aucune tuile valide")
            return
    
    # 4. Traiter les tuiles valides
        return self._traiter_tuiles_selectionnees(tuiles_valides, dossier_sortie)
        
    def traitement_batch_tuiles(self, nb_tuiles=100, dossier_sortie="comp_RPG", 
                               min_parcelles_rpg=1, seed=42):
        """
        Traite un échantillon aléatoire de tuiles et sauvegarde les visualisations
        
        Parameters:
        -----------
        nb_tuiles : int
            Nombre de tuiles à traiter
        dossier_sortie : str  
            Nom du dossier de sortie
        min_parcelles_rpg : int
            Nombre minimum de parcelles RPG pour inclure une tuile
        seed : int
            Seed pour la reproductibilité
        """
        
        import os
        import random
        
        # Créer le dossier de sortie
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
            print(f"Dossier créé: {dossier_sortie}")
        
        # Obtenir toutes les tuiles disponibles
        print("Récupération de toutes les tuiles disponibles...")
        tuiles_disponibles = self.lister_tuiles_disponibles()
        
        if tuiles_disponibles.empty:
            print("❌ Aucune tuile disponible")
            return
        
        print(f"Total tuiles disponibles: {len(tuiles_disponibles)}")
        
        # Filtrer les tuiles avec suffisamment de parcelles RPG
        tuiles_valides = []
        
        print("Filtrage des tuiles avec parcelles RPG...")
        for idx, (_, tuile_info) in enumerate(tuiles_disponibles.iterrows()):
            if idx % 50 == 0:
                print(f"  Progression: {idx}/{len(tuiles_disponibles)}")
                
            tuile_id = tuile_info['tuile_id']
            coords_info = self.obtenir_coords_tuile(tuile_id)
            
            if coords_info:
                rpg_filtre = self.filtrer_rpg_par_tuile(coords_info['geometry'])
                
                if len(rpg_filtre) >= min_parcelles_rpg:
                    tuiles_valides.append({
                        'tuile_id': tuile_id,
                        'fold': tuile_info['fold'],
                        'nb_parcelles': len(rpg_filtre),
                        'coords_info': coords_info
                    })
        
        print(f"Tuiles valides (≥{min_parcelles_rpg} parcelles): {len(tuiles_valides)}")
        
        if len(tuiles_valides) == 0:
            print("❌ Aucune tuile avec parcelles RPG trouvée")
            return
        
        # Échantillonnage aléatoire
        random.seed(seed)
        nb_a_traiter = min(nb_tuiles, len(tuiles_valides))
        tuiles_selectionnees = random.sample(tuiles_valides, nb_a_traiter)
        
        print(f"Tuiles sélectionnées pour traitement: {nb_a_traiter}")
        
        # Statistiques des tuiles sélectionnées
        parcelles_stats = [t['nb_parcelles'] for t in tuiles_selectionnees]
        print(f"Statistiques parcelles RPG:")
        print(f"  Moyenne: {np.mean(parcelles_stats):.1f}")
        print(f"  Min: {np.min(parcelles_stats)}")
        print(f"  Max: {np.max(parcelles_stats)}")
        
        # Traitement des tuiles
        resultats = []
        succes = 0
        echecs = 0
        
        print(f"\nDébut du traitement...")
        print("=" * 60)
        
        for idx, tuile_data in enumerate(tuiles_selectionnees):
            tuile_id = tuile_data['tuile_id']
            fold = tuile_data['fold']
            
            print(f"[{idx+1}/{nb_a_traiter}] Traitement tuile {tuile_id}...")
            
            try:
                # Visualiser la tuile
                fig, data = self.visualiser_tuile_complete(
                    tuile_id, fold, figsize=(24, 12)
                )
                
                if fig is not None:
                    # Nom du fichier de sortie
                    nom_fichier = f"tuile_{tuile_id}_fold_{fold}_parcelles_{tuile_data['nb_parcelles']}.png"
                    chemin_complet = os.path.join(dossier_sortie, nom_fichier)
                    
                    # Sauvegarder
                    fig.savefig(chemin_complet, dpi=200, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    
                    print(f"  ✅ Sauvegardé: {nom_fichier}")
                    
                    # Fermer la figure pour libérer la mémoire
                    plt.close(fig)
                    
                    # Collecter les statistiques
                    resultats.append({
                        'tuile_id': tuile_id,
                        'fold': fold,
                        'nb_parcelles': tuile_data['nb_parcelles'],
                        'fichier': nom_fichier,
                        'statut': 'succès'
                    })
                    
                    succes += 1
                    
                else:
                    print(f"  ❌ Échec: figure non générée")
                    echecs += 1
                    
            except Exception as e:
                print(f"  ❌ Erreur: {str(e)}")
                echecs += 1
                
                resultats.append({
                    'tuile_id': tuile_id,
                    'fold': fold,
                    'nb_parcelles': tuile_data['nb_parcelles'],
                    'fichier': None,
                    'statut': f'erreur: {str(e)}'
                })
        
        # Résumé final
        print("\n" + "=" * 60)
        print("RÉSUMÉ DU TRAITEMENT")
        print("=" * 60)
        print(f"Tuiles traitées: {nb_a_traiter}")
        print(f"Succès: {succes}")
        print(f"Échecs: {echecs}")
        print(f"Taux de succès: {succes/nb_a_traiter*100:.1f}%")
        print(f"Images sauvegardées dans: {dossier_sortie}/")
        
        # Sauvegarder un rapport
        self._sauvegarder_rapport_batch(resultats, dossier_sortie, tuiles_selectionnees)
        
        return resultats

    def _sauvegarder_rapport_batch(self, resultats, dossier_sortie, tuiles_selectionnees):
        """Sauvegarde un rapport détaillé du traitement"""
        
        import json
        from datetime import datetime
        
        rapport = {
            'date_traitement': datetime.now().isoformat(),
            'nb_tuiles_traitees': len(resultats),
            'nb_succes': len([r for r in resultats if r['statut'] == 'succès']),
            'nb_echecs': len([r for r in resultats if r['statut'] != 'succès']),
            'statistiques_parcelles': {
                'moyenne': np.mean([t['nb_parcelles'] for t in tuiles_selectionnees]),
                'min': np.min([t['nb_parcelles'] for t in tuiles_selectionnees]),
                'max': np.max([t['nb_parcelles'] for t in tuiles_selectionnees])
            },
            'tuiles_traitees': resultats
        }
        
        # Sauvegarder en JSON
        fichier_rapport = os.path.join(dossier_sortie, 'rapport_traitement.json')
        with open(fichier_rapport, 'w', encoding='utf-8') as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder aussi en texte lisible
        fichier_txt = os.path.join(dossier_sortie, 'rapport_traitement.txt')
        with open(fichier_txt, 'w', encoding='utf-8') as f:
            f.write(f"RAPPORT DE TRAITEMENT BATCH\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tuiles traitées: {len(resultats)}\n")
            f.write(f"Succès: {rapport['nb_succes']}\n")
            f.write(f"Échecs: {rapport['nb_echecs']}\n")
            f.write(f"Taux de succès: {rapport['nb_succes']/len(resultats)*100:.1f}%\n\n")
            f.write(f"Statistiques parcelles RPG:\n")
            f.write(f"  Moyenne: {rapport['statistiques_parcelles']['moyenne']:.1f}\n")
            f.write(f"  Min: {rapport['statistiques_parcelles']['min']}\n")
            f.write(f"  Max: {rapport['statistiques_parcelles']['max']}\n\n")
            f.write("DÉTAIL DES TUILES:\n")
            f.write("-" * 50 + "\n")
            
            for r in resultats:
                f.write(f"Tuile {r['tuile_id']} (fold {r['fold']}): ")
                f.write(f"{r['nb_parcelles']} parcelles - {r['statut']}\n")
        
        print(f"Rapport sauvegardé: {fichier_rapport}")
        print(f"Rapport texte: {fichier_txt}")
    def _afficher_confidence(self, ax, confidence, titre):
        """Affiche la carte de confiance"""
    # Vérifier si confidence est un array numpy
        if isinstance(confidence, np.ndarray):
            if len(confidence.shape) >= 3:
                if confidence.shape[0] == 1:
                    confidence = confidence[0]
                else:
                    confidence = np.mean(confidence, axis=0)
        
            im = ax.imshow(confidence, cmap='viridis')
            ax.set_title(titre)
            plt.colorbar(im, ax=ax, label='Confiance')
        else:
        # Si confidence est un scalaire, afficher juste la valeur
            ax.text(0.5, 0.5, f'Confiance moyenne:\n{confidence:.3f}', 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=12)
            ax.set_title(titre)
            ax.axis('off')

    def _geo_vers_pixel(self, gdf, coords_info, shape_image):
        """Convertit les coordonnées géographiques en coordonnées pixel"""
        height, width = shape_image
        xmin, ymin = coords_info['xmin'], coords_info['ymin']
        xmax, ymax = coords_info['xmax'], coords_info['ymax']
        
        def transform_coords(coords):
            pixel_coords = []
            for x, y in coords:
                pixel_x = (x - xmin) / (xmax - xmin) * width
                pixel_y = (ymax - y) / (ymax - ymin) * height
                pixel_coords.append((pixel_x, pixel_y))
            return pixel_coords
        
        transformed_geoms = []
        for geom in gdf.geometry:
            if geom.geom_type == 'Polygon':
                exterior_coords = transform_coords(list(geom.exterior.coords))
                from shapely.geometry import Polygon as ShapelyPolygon
                transformed_geoms.append(ShapelyPolygon(exterior_coords))
            elif geom.geom_type == 'MultiPolygon':
                largest_poly = max(geom.geoms, key=lambda p: p.area)
                exterior_coords = transform_coords(list(largest_poly.exterior.coords))
                from shapely.geometry import Polygon as ShapelyPolygon
                transformed_geoms.append(ShapelyPolygon(exterior_coords))
        
        return gpd.GeoDataFrame(gdf.drop('geometry', axis=1), 
                              geometry=transformed_geoms)

    def _dessiner_polygones_pixel(self, ax, gdf_pixel, color='red', linewidth=2):
        """Dessine les polygones en coordonnées pixel"""
        for _, row in gdf_pixel.iterrows():
            if hasattr(row.geometry, 'exterior'):
                coords = list(row.geometry.exterior.coords)
                polygon = Polygon(coords, fill=False, edgecolor=color, linewidth=linewidth)
                ax.add_patch(polygon)
# Fonction utilitaire pour explorer les données
def explorer_structure_tuile(visualisateur, tuile_id, fold='0'):
    """Explore la structure d'une tuile"""
    chemin_s3 = f"{visualisateur.bucket_name}/{tuile_id}_best_fold_{fold}.npy"
    data = visualisateur.charger_tuile_depuis_s3(chemin_s3)
    
    print(f"Structure de la tuile {tuile_id}:")
    print(f"Clés disponibles: {list(data.keys())}")
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            if key in ['pano_semantic', 'pano_instance']:
                print(f"  - Valeurs uniques: {len(np.unique(value))}")
                print(f"  - Min/Max: {value.min()}/{value.max()}")
        else:
            print(f"{key}: type {type(value)}")
    
    return data

def verifier_geojson_tuiles(chemin_geojson):
    """Vérifie la structure du fichier GeoJSON des tuiles"""
    import geopandas as gpd
    
    # Charger le GeoJSON
    tuiles_gdf = gpd.read_file(chemin_geojson)
    
    print(f"Fichier GeoJSON chargé:")
    print(f"- Nombre de tuiles: {len(tuiles_gdf)}")
    print(f"- CRS: {tuiles_gdf.crs}")
    print(f"- Colonnes: {list(tuiles_gdf.columns)}")
    print(f"- Premiers IDs: {sorted(tuiles_gdf['id'].unique())[:10]}")
    
    # Vérifier les géométries
    print(f"- Types de géométries: {tuiles_gdf.geometry.geom_type.unique()}")
    
    # Exemple de bounds pour les premières tuiles
    print("\nExemples de bounds (xmin, ymin, xmax, ymax):")
    for i in range(min(3, len(tuiles_gdf))):
        row = tuiles_gdf.iloc[i]
        bounds = row.geometry.bounds
        print(f"Tuile ID {row['id']}: {bounds}")
    
    return tuiles_gdf

def exemple_utilisation():
    """Exemple d'utilisation avec les bons paramètres"""
    
    # Paramètres adaptés à votre structure
    BUCKET_NAME = "antoinelesauvage"
    SOUS_DOSSIER = "vergers-france/preds_128_10"
    CHEMIN_RPG = "PARCELLES_GRAPHIQUES.shp"  # À ADAPTER
    CHEMIN_GEOJSON_TUILES = "grid_indre_loire_128_lambert93.geojson"
    
    try:
        # Créer le visualisateur
        visualisateur = VisualisateurSegmentationRPG_Onyxia(
            BUCKET_NAME,
            SOUS_DOSSIER,
            CHEMIN_RPG, 
            CHEMIN_GEOJSON_TUILES
        )
        
        # Lister les tuiles disponibles (limiter pour test)
        print("Recherche des tuiles avec coordonnées...")
        tuiles_disponibles = visualisateur.lister_tuiles_disponibles(limite=100)  # Test avec 100 premières
        
        print(f"Tuiles disponibles avec coordonnées: {len(tuiles_disponibles)}")
        
        if len(tuiles_disponibles) > 0:
            print("Premières tuiles trouvées:")
            print(tuiles_disponibles[['tuile_id', 'fold', 'xmin', 'ymin', 'xmax', 'ymax']].head())
            
            # Explorer la structure d'une tuile
            premiere_tuile = tuiles_disponibles.iloc[10]
            print(f"\nTest de visualisation de la tuile {premiere_tuile['tuile_id']}...")
            
            fig, data = visualisateur.visualiser_tuile_complete(
                premiere_tuile['tuile_id'],
                premiere_tuile['fold']
            )
            
            if fig is not None:
                plt.show()
                
        else:
            print("Aucune tuile trouvée avec coordonnées correspondantes")
            
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


# Test rapide de la connexion S3
def test_rapide_s3():
    """Test rapide pour vérifier la connexion S3 et charger une tuile"""
    import s3fs
    
    # Configuration S3 Onyxia
    fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])
    
    # Lister les fichiers dans votre bucket
    bucket_name = "antoinelesauvage"  # À remplacer
    files = fs.glob(f"{bucket_name}/vergers-france/preds_128_10/*_best_fold_*.npy")
    print(f"Fichiers trouvés: {len(files)}")
    print("Premiers fichiers:", files[:5])
    
    if files:
        # Tester le chargement d'un fichier
        test_file = files[0]
        print(f"\nTest de chargement: {test_file}")
        
        with fs.open(test_file, 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
            print(f"Clés: {list(data.keys())}")
            
            # Vérifier la structure
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: {value.shape}")

def lancer_batch_simple():
    """Fonction simple pour lancer le traitement batch"""
    
    # Paramètres
    BUCKET_NAME = "antoinelesauvage"
    SOUS_DOSSIER = "vergers-france/preds_128_10"
    CHEMIN_RPG = "PARCELLES_GRAPHIQUES.shp"  # À ADAPTER
    CHEMIN_GEOJSON_TUILES = "grid_indre_loire_128_lambert93.geojson"
    
    try:
        # Créer le visualisateur
        visualisateur = VisualisateurSegmentationRPG_Onyxia(
            BUCKET_NAME,
            SOUS_DOSSIER,
            CHEMIN_RPG, 
            CHEMIN_GEOJSON_TUILES
        )
        
        # Lancer le traitement batch
        resultats = visualisateur.traitement_batch_rapide(
            nb_tuiles=100,
            dossier_sortie="comp_RPG",
            min_parcelles_rpg=1,  # Inclure même les tuiles avec 1 seule parcelle
            seed=42
        )
        
        print("✅ Traitement batch terminé!")
        
        return resultats
        
    except Exception as e:
        print(f"❌ Erreur dans le traitement batch: {e}")
        import traceback
        traceback.print_exc()
        return None

# Pour lancer
if __name__ == "__main__":
    resultats = lancer_batch_simple()