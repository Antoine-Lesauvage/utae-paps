import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from collections import defaultdict
import boto3
import s3fs
import pickle
from io import BytesIO
import os
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes
from shapely.geometry import shape
os.environ["AWS_ACCESS_KEY_ID"] = 'BS0G2KO2FTSK62F36PRV'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'lXtVTVZr29H+aP+NNtWxnJv9g1zVkvwD0k5kkJ1M'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJCUzBHMktPMkZUU0s2MkYzNlBSViIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU2MzY2MDAwLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU2OTcyNTkzLCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NjM2Nzc5MiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDo5OGUwY2U5Mi1jZmRkLTA0ZGEtM2FhNi0yMDQxZTgyNzYzZDMiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJlNDYxYTBkYS0wNzkxLTRmYTctOThmMi0wNGFmZjZhMGY2NDMiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.p1E_oPr45DsLUk8BNnApG8__huhL3UtKHdoxyv3Nle0p4jL6atUm964C3RS9OAay7BO9DPWtrtV6WPuxmddxHA'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
class PanopticS3Evaluator:
    def __init__(self, rpg_path, s3_config=None):
        """
        Args:
            rpg_path: chemin vers le RPG
            s3_config: dict avec les config S3 ou None pour auto-detection Onyxia
        """
        self.rpg = gpd.read_file(rpg_path)
        
        # Configuration S3 pour Onyxia
        if s3_config is None:
            self.s3_config = self._get_onyxia_s3_config()
        else:
            self.s3_config = s3_config
        
        # Initialiser le système de fichiers S3 avec le token
        self.fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': self.s3_config['endpoint_url']},
            key=self.s3_config['access_key'],
            secret=self.s3_config['secret_key'],
            token=self.s3_config.get('session_token')
        )
        
        # Initialiser le client boto3 avec le token
        session = boto3.Session(
            aws_access_key_id=self.s3_config['access_key'],
            aws_secret_access_key=self.s3_config['secret_key'],
            aws_session_token=self.s3_config.get('session_token'),
            region_name=self.s3_config.get('region', 'us-east-1')
        )
        
        self.s3_client = session.client(
            's3',
            endpoint_url=self.s3_config['endpoint_url']
        )
        
        # Mappings de classes
        self.class_mapping = {
            0: 'Arrière-plan', 1: 'Prairie', 2: 'Blé tendre hiver',
            3: 'Maïs', 4: 'Orge hiver', 5: 'Colza hiver',
            6: 'Orge printemps', 7: 'Tournesol', 8: 'Vigne',
            9: 'Betterave', 10: 'Triticale hiver', 11: 'Blé dur hiver',
            12: 'Fruits/légumes/fleurs', 13: 'Pommes de terre',
            14: 'Fourrage légumineux', 15: 'Soja', 16: 'Verger',
            17: 'Céréale mélangée', 18: 'Sorgho', 19: 'Étiquette vide'
        }
        
        self.culture_mapping = {
            "Blé tendre d'hiver": [('BTH', [1])],
            "Blé dur d'hiver": [('BDH', [1])],
            'Maïs': [(None, [2])],
            "Orge d'hiver": [('ORH',[3])],
            "Orge de printemps": [('ORP',[3])],
            "Triticale d'hiver": [("TTH", [4])],
            'Sorgho': [('SOG', [4])],
            "Céréales mélangées": [('MCR', [4])],
            "Colza d'hiver": [(None, [5])],
            "Tournesol": [(None, [6])],
            "Soja": [("SOJ", [7])],
            "Prairie": [(None, [18,19])],
            "Verger": [(None, [20])],
            "Vigne": [(None, [10])],
            "Betterave": [("BVF", [16]),("BTN",[24])],
            "Fruits, légumes, fleurs": [(None, [15,22,24,25])],
            "Fourrage légumineux": [("MLG", [16])],
            "Pommes de terre": [("PTC", [25])]
        }
        
        # Préparer le RPG
        if self.rpg.crs != 'EPSG:2154':
            self.rpg = self.rpg.to_crs('EPSG:2154')
        
        self.rpg = self.creer_correspondance_rpg(self.rpg, self.culture_mapping)
        print(f"RPG chargé avec {len(self.rpg)} parcelles")

    def _get_onyxia_s3_config(self):
        """Récupère la config S3 d'Onyxia depuis les variables d'environnement"""
        
        config = {
            'endpoint_url': 'https://minio.lab.sspcloud.fr',
            'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'session_token': os.getenv('AWS_SESSION_TOKEN'),
            'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        }
        
        if not config['access_key'] or not config['secret_key']:
            raise ValueError(
                "Credentials S3 non trouvées. Définissez AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY"
            )
        
        return config

    def creer_correspondance_rpg(self, gdf_rpg, culture_mapping):
        """Crée une correspondance entre les données RPG et les classes de culture"""
        gdf_rpg = gdf_rpg.copy()
        gdf_rpg['classe_harmonisee'] = None
        
        # Convertir CODE_GROUP en numérique
        gdf_rpg['CODE_GROUP'] = pd.to_numeric(gdf_rpg['CODE_GROUP'], errors='coerce')
        gdf_rpg['CODE_CULTU'] = gdf_rpg['CODE_CULTU'].astype(str).str.strip()
        
        print("Début du mapping RPG...")
        
        for classe_pred, conditions in culture_mapping.items():
            mask = pd.Series([False] * len(gdf_rpg), index=gdf_rpg.index)
            
            for code_cultu, codes_group in conditions:
                if code_cultu is not None:
                    mask_cultu = gdf_rpg['CODE_CULTU'] == code_cultu
                else:
                    mask_cultu = pd.Series([True] * len(gdf_rpg), index=gdf_rpg.index)
                
                mask_group = gdf_rpg['CODE_GROUP'].isin(codes_group)
                mask_combined = mask_cultu & mask_group
                mask = mask | mask_combined
            
            gdf_rpg.loc[mask, 'classe_harmonisee'] = classe_pred
        
        return gdf_rpg

    def semantic_class_to_crop_name(self, semantic_class):
        """Convertit une classe sémantique en nom de culture harmonisé"""
        class_name = self.class_mapping.get(semantic_class, f'unknown_{semantic_class}')
        
        correspondance = {
            'Prairie': 'Prairie',
            'Blé tendre hiver': "Blé tendre d'hiver", 
            'Maïs': 'Maïs',
            'Orge hiver': "Orge d'hiver",
            'Colza hiver': "Colza d'hiver",
            'Orge printemps': "Orge de printemps",
            'Tournesol': 'Tournesol',
            'Vigne': 'Vigne',
            'Betterave': 'Betterave',
            'Triticale hiver': "Triticale d'hiver",
            'Blé dur hiver': "Blé dur d'hiver",
            'Fruits/légumes/fleurs': "Fruits, légumes, fleurs",
            'Pommes de terre': "Pommes de terre",
            'Fourrage légumineux': "Fourrage légumineux",
            'Soja': 'Soja',
            'Verger': 'Verger',
            'Céréale mélangée': "Céréales mélangées",
            'Sorgho': 'Sorgho',
            'Étiquette vide': None,
            'Arrière-plan': None
        }
        
        return correspondance.get(class_name, class_name)

    def list_tiles_in_s3(self, bucket_name, prefix=""):
        """Liste toutes les tuiles disponibles dans le bucket S3"""
        try:
            # Construire le chemin complet
            if prefix:
                # Retirer le nom du bucket du prefix s'il y est
                if prefix.startswith(f"{bucket_name}/"):
                    prefix = prefix.replace(f"{bucket_name}/", "", 1)
                full_path = f's3://{bucket_name}/{prefix}'
            else:
                full_path = f's3://{bucket_name}'
            
            print(f"Listage de: {full_path}")
            
            objects = self.fs.ls(full_path, detail=False)
            
            tile_files = []
            for obj in objects:
                # Nettoyer le chemin correctement
                if obj.startswith(f's3://{bucket_name}/'):
                    clean_path = obj.replace(f's3://{bucket_name}/', '', 1)
                else:
                    clean_path = obj
                
                # Retirer le nom du bucket s'il apparaît au début du chemin
                if clean_path.startswith(f"{bucket_name}/"):
                    clean_path = clean_path.replace(f"{bucket_name}/", "", 1)
                
                filename = clean_path.split('/')[-1]
                
                # Vérifier l'extension
                if any(filename.endswith(ext) for ext in ['.pkl', '.pickle', '.npy', '.npz', '.json']):
                    tile_files.append(clean_path)
            
            print(f"Trouvé {len(tile_files)} fichiers de tuiles")
            if tile_files:
                print("Exemples de fichiers (chemins corrigés):")
                for file in tile_files[:5]:
                    print(f"  - {file}")
            
            return tile_files
            
        except Exception as e:
            print(f"Erreur lors de la liste des objets S3: {e}")
            return []

    def load_tile_data_from_s3(self, bucket_name, tile_key):
        """Charge les données d'une tuile depuis S3"""
        try:
            # Construire le chemin S3 correct
            s3_path = f's3://{bucket_name}/{tile_key}'
            
            print(f"Tentative de chargement: {s3_path}")
            
            # Méthode pour fichiers numpy
            if tile_key.endswith('.npy'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                return data
            
            elif tile_key.endswith('.npz'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                return data
            
            # Méthode pour fichiers pickle
            elif tile_key.endswith('.pkl') or tile_key.endswith('.pickle'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = pickle.load(f)
                return data
            
            # Méthode pour JSON
            elif tile_key.endswith('.json'):
                import json
                with self.fs.open(s3_path, 'r') as f:
                    data = json.load(f)
                return data
            
            else:
                print(f"Format de fichier non supporté: {tile_key}")
                return None
                
        except Exception as e:
            print(f"Erreur lors du chargement de {tile_key}: {e}")
            return None

    def extract_tile_id_from_key(self, tile_key):
        """Extrait l'ID de la tuile depuis la clé S3"""
        
        # Retirer le nom du bucket s'il est présent au début
        if tile_key.startswith("antoinelesauvage/"):
            tile_key = tile_key.replace("antoinelesauvage/", "", 1)
        
        filename = tile_key.split('/')[-1]
        name_without_ext = filename.split('.')[0]
        
        # Pour vos fichiers : 851_best_fold_5.npy -> 851
        import re
        match = re.search(r'(\d+)_best_fold_\d+', name_without_ext)
        if match:
            return match.group(1)
        
        # Fallback
        return name_without_ext.split('_')[0] if '_' in name_without_ext else name_without_ext

    def get_tile_geometry_from_geojson(self, tile_id, tiles_geojson_path):
        """Récupère la géométrie d'une tuile depuis le GeoJSON"""
        tiles_gdf = gpd.read_file(tiles_geojson_path)
        
        # Essayer différentes colonnes possibles pour l'ID
        id_columns = ['tile_id', 'id', 'ID', 'name', 'NAME']
        
        tile_geom = None
        for col in id_columns:
            if col in tiles_gdf.columns:
                matching_tiles = tiles_gdf[tiles_gdf[col].astype(str) == str(tile_id)]
                if not matching_tiles.empty:
                    tile_geom = matching_tiles.geometry.iloc[0]
                    break
        
        if tile_geom is None:
            print(f"Géométrie non trouvée pour tile_id: {tile_id}")
            return None
        
        # Assurer le bon CRS
        if tiles_gdf.crs != 'EPSG:2154':
            tile_geom = gpd.GeoSeries([tile_geom], crs=tiles_gdf.crs).to_crs('EPSG:2154').iloc[0]
        
        return tile_geom
    def evaluate_tiles_from_s3(self, bucket_name, tiles_geojson_path, 
                           prefix="", confidence_threshold=0.5, 
                           max_tiles=None, grid_crs='EPSG:2154'):
        """
        Évalue toutes les tuiles depuis S3
    
        Args:
            bucket_name: nom du bucket S3
            tiles_geojson_path: chemin vers le GeoJSON des tuiles
            prefix: préfixe S3 pour filtrer les tuiles
            confidence_threshold: seuil de confiance minimum
            max_tiles: nombre maximum de tuiles à traiter (pour test)
            grid_crs: CRS de la grille des tuiles
        """
    # Correction du prefix
        if prefix.startswith(f"{bucket_name}/"):
            prefix = prefix.replace(f"{bucket_name}/", "", 1)
            print(f"Prefix corrigé: {prefix}")
    
    # Lister les tuiles dans S3
        tile_files = self.list_tiles_in_s3(bucket_name, prefix)
    
        if max_tiles:
            tile_files = tile_files[:max_tiles]
            print(f"Limitation à {max_tiles} tuiles pour test")
    
        all_results = {}
        detailed_analysis = []
        processed_tiles = 0
    
        for tile_key in tile_files:
            print(f"\n--- Traitement de {tile_key} ({processed_tiles+1}/{len(tile_files)}) ---")
        
        # Extraire l'ID de la tuile
            tile_id = self.extract_tile_id_from_key(tile_key)
        
        # Charger les données depuis S3
            tile_data = self.load_tile_data_from_s3(bucket_name, tile_key)
            if tile_data is None:
                print(f"Impossible de charger {tile_key}")
                continue
        
        # Récupérer la géométrie de la tuile
            tile_geometry = self.get_tile_geometry_from_geojson(tile_id, tiles_geojson_path)
            if tile_geometry is None:
                print(f"Géométrie non trouvée pour {tile_id}")
                continue
        
        # Extraire les instances panoptiques
            try:
                instances = self.extract_instances_from_tile_data(
                tile_data, tile_id, tile_geometry, grid_crs
                )
            except Exception as e:
                print(f"Erreur extraction instances pour {tile_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
            if not instances:
                print(f"Aucune instance extraite pour {tile_id}")
                continue
        
            print(f"Instances extraites: {len(instances)}")
        
        # Évaluer cette tuile
            try:
                tile_results = self.evaluate_tile_instances(
                    instances, tile_geometry, confidence_threshold
                )
            
                if tile_results:
                    all_results[tile_id] = tile_results
                    processed_tiles += 1
                
                # Analyse détaillée des instances
                    pred_gdf = self.create_instances_gdf(instances, confidence_threshold)
                    if not pred_gdf.empty:
                        tile_bounds = tile_geometry.bounds
                        rpg_tile = self.rpg.cx[
                            tile_bounds[0]-100:tile_bounds[2]+100,
                            tile_bounds[1]-100:tile_bounds[3]+100
                        ]
                        rpg_tile = rpg_tile[rpg_tile['classe_harmonisee'].notna()]
                    
                        if not rpg_tile.empty:
                            instance_analysis = self.analyze_instance_quality(pred_gdf, rpg_tile)
                            instance_analysis['tile_id'] = tile_id
                            detailed_analysis.append(instance_analysis)
                else:
                    print(f"Aucun résultat d'évaluation pour {tile_id}")
            
            except Exception as e:
                print(f"Erreur évaluation pour {tile_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
        print(f"\n=== TRAITEMENT TERMINÉ ===")
        print(f"Tuiles traitées avec succès: {processed_tiles}/{len(tile_files)}")
    
    # Consolidation des résultats
        detailed_df = pd.concat(detailed_analysis, ignore_index=True) if detailed_analysis else pd.DataFrame()
    
        return all_results, detailed_df
    def list_tiles_in_s3(self, bucket_name, prefix=""):
        """Liste toutes les tuiles disponibles dans le bucket S3"""
        try:
        # Construire le chemin complet
            if prefix:
            # Retirer le nom du bucket du prefix s'il y est
                if prefix.startswith(f"{bucket_name}/"):
                    prefix = prefix.replace(f"{bucket_name}/", "", 1)
                full_path = f's3://{bucket_name}/{prefix}'
            else:
                full_path = f's3://{bucket_name}'
        
            print(f"Listage de: {full_path}")
        
            objects = self.fs.ls(full_path, detail=False)
        
            tile_files = []
            for obj in objects:
            # Nettoyer le chemin correctement
                if obj.startswith(f's3://{bucket_name}/'):
                    clean_path = obj.replace(f's3://{bucket_name}/', '', 1)
                else:
                    clean_path = obj
            
            # Retirer le nom du bucket s'il apparaît au début du chemin
                if clean_path.startswith(f"{bucket_name}/"):
                    clean_path = clean_path.replace(f"{bucket_name}/", "", 1)
            
                filename = clean_path.split('/')[-1]
            
            # Vérifier l'extension
                if any(filename.endswith(ext) for ext in ['.pkl', '.pickle', '.npy', '.npz', '.json']):
                    tile_files.append(clean_path)
        
            print(f"Trouvé {len(tile_files)} fichiers de tuiles")
            if tile_files:
                print("Exemples de fichiers (chemins corrigés):")
                for file in tile_files[:5]:
                    print(f"  - {file}")
        
            return tile_files     
        except Exception as e:
            print(f"Erreur lors de la liste des objets S3: {e}")
            return []
    def get_tile_transform(self, tile_geometry, raster_shape):
        """Calcule la transformation géographique pour une tuile"""
        bounds = tile_geometry.bounds
        minx, miny, maxx, maxy = bounds
        
        height, width = raster_shape
        
        # Calculer la résolution
        res_x = (maxx - minx) / width
        res_y = (maxy - miny) / height
        
        # Créer la transformation affine
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        return transform

    def mask_to_polygons_georef(self, mask, transform):
        """Convertit un masque en polygones géoréférencés"""
        polygons = []
        
        # Extraire les formes du masque
        for geom, value in shapes(mask.astype(np.uint8), transform=transform):
            if value == 1:  # Seulement les pixels avec valeur 1
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
        
        return polygons

    def calculate_box_iou(self, box1, box2):
        """Calcule l'IoU entre deux bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Aires des boîtes
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # IoU
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def extract_instances_from_tile_data(self, data, tile_id, tile_geometry, grid_crs):
        """Extraction avec géoréférencement correct en coordonnées métriques"""
        if data is None:
            return None
        
        # Si c'est un dictionnaire
        if isinstance(data, dict):
            pano_instance = data.get('pano_instance')
            pano_semantic = data.get('pano_semantic')
            instance_boxes = data.get('instance_boxes')
            confidence = data.get('confidence', None)
        else:
            try:
                data_dict = data.item()
                pano_instance = data_dict.get('pano_instance')
                pano_semantic = data_dict.get('pano_semantic')
                instance_boxes = data_dict.get('instance_boxes')
                confidence = data_dict.get('confidence', None)
            except:
                return None
        
        if pano_instance is None or pano_semantic is None:
            return None
        
        # Traiter les dimensions
        if len(pano_instance.shape) == 3 and pano_instance.shape[0] == 1:
            pano_instance = pano_instance.squeeze(0)
        
        if len(pano_semantic.shape) == 4:
            if pano_semantic.shape[0] == 1:
                pano_semantic = pano_semantic.squeeze(0)
            pano_semantic = np.argmax(pano_semantic, axis=0)
        elif len(pano_semantic.shape) == 3 and pano_semantic.shape[0] == 1:
            pano_semantic = pano_semantic.squeeze(0)
        
        # Confidence scores
        confidence_scores = {}
        if confidence is not None and instance_boxes is not None:
            if isinstance(confidence, (int, float)):
                confidence_flat = [confidence]
            elif hasattr(confidence, 'flatten'):
                confidence_flat = confidence.flatten()
            elif isinstance(confidence, (list, tuple)):
                confidence_flat = confidence
            else:
                confidence_flat = []
            
            if hasattr(instance_boxes, 'cpu'):
                boxes = instance_boxes.cpu().numpy()
            else:
                boxes = np.array(instance_boxes)
            
            unique_instances = np.unique(pano_instance)[1:]
            
            for idx, instance_id in enumerate(unique_instances):
                instance_mask = (pano_instance == instance_id)
                y_indices, x_indices = np.where(instance_mask)
                
                if len(y_indices) == 0:
                    continue
                
                if len(boxes) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    instance_box = [x_min, y_min, x_max, y_max]
                    
                    best_match_idx = None
                    best_iou = 0
                    
                    for i, box in enumerate(boxes):
                        pred_box = box
                        iou = self.calculate_box_iou(instance_box, pred_box)
                        
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            best_match_idx = i
                    
                    if best_match_idx is not None and best_match_idx < len(confidence_flat):
                        confidence_scores[instance_id] = float(confidence_flat[best_match_idx])
                else:
                    if len(confidence_flat) == 1:
                        confidence_scores[instance_id] = float(confidence_flat[0])
                    elif idx < len(confidence_flat):
                        confidence_scores[instance_id] = float(confidence_flat[idx])
        
        # Calculer la transformation géographique
        transform = self.get_tile_transform(tile_geometry, pano_instance.shape)
        
        # Traiter chaque instance
        results = []
        unique_instances = np.unique(pano_instance)[1:]
        
        for instance_id in unique_instances:
            mask = (pano_instance == instance_id).astype(np.uint8)
            
            # Classe sémantique majoritaire
            instance_pixels = pano_instance == instance_id
            semantic_values = pano_semantic[instance_pixels]
            semantic_values_nonzero = semantic_values[semantic_values > 0]
            
            if len(semantic_values_nonzero) > 0:
                semantic_class = np.bincount(semantic_values_nonzero).argmax()
            else:
                semantic_class = np.bincount(semantic_values).argmax()
            
            # Convertir en polygones
            try:
                polygons = self.mask_to_polygons_georef(mask, transform)
            except Exception as e:
                continue
            
            if len(polygons) == 0:
                continue
            
            conf_score = confidence_scores.get(instance_id, None)
            
            for i, poly in enumerate(polygons):
                if poly.area > 0:
                    result = {
                        'tile_id': tile_id,
                        'instance_id': int(instance_id),
                        'polygon_id': i,
                        'semantic_class': int(semantic_class),
                        'geometry': poly,
                        'area_pixels': np.sum(mask),
                        'area_m2': poly.area,
                        'perimeter_m': poly.length,
                        'centroid_x': poly.centroid.x,
                        'centroid_y': poly.centroid.y
                    }
                    
                    if conf_score is not None:
                        result['confidence'] = conf_score
                    
                    results.append(result)
        
        return results

    def create_instances_gdf(self, instances_results, confidence_threshold=0.5):
        """Convertit les résultats d'instances en GeoDataFrame avec classes harmonisées"""
        
        if not instances_results:
            return gpd.GeoDataFrame()
        
        filtered_instances = []
        for instance in instances_results:
            # Filtrer par confiance si disponible
            if 'confidence' in instance and instance['confidence'] < confidence_threshold:
                continue
            
            # Ignorer l'arrière-plan
            if instance['semantic_class'] == 0:
                continue
            
            # Mapper vers le nom de culture harmonisé
            crop_type = self.semantic_class_to_crop_name(instance['semantic_class'])
            
            # Ignorer les classes non mappées
            if crop_type is None:
                continue
            
            filtered_instances.append({
                'instance_id': instance['instance_id'],
                'semantic_class': instance['semantic_class'],
                'crop_type': crop_type,
                'geometry': instance['geometry'],
                'area_m2': instance['area_m2'],
                'confidence': instance.get('confidence', None)
            })
        
        if not filtered_instances:
            return gpd.GeoDataFrame()
        
        gdf = gpd.GeoDataFrame(filtered_instances)
        gdf.crs = 'EPSG:2154'
        return gdf

    def evaluate_tile_instances(self, instances_results, tile_geometry, confidence_threshold=0.5):
        """Évalue les instances d'une tuile contre le RPG"""
        
        if not instances_results:
            return {}
        
        # Créer GeoDataFrame des prédictions
        pred_gdf = self.create_instances_gdf(instances_results, confidence_threshold)
        if pred_gdf.empty:
            return {}
        
        # Filtrer le RPG pour cette tuile
        tile_bounds = tile_geometry.bounds
        buffer = 100  # 100m de buffer
        
        rpg_tile = self.rpg.cx[
            tile_bounds[0]-buffer:tile_bounds[2]+buffer,
            tile_bounds[1]-buffer:tile_bounds[3]+buffer
        ].copy()
        
        # Filtrer seulement les parcelles avec une classe harmonisée
        rpg_tile = rpg_tile[rpg_tile['classe_harmonisee'].notna()].copy()
        
        if rpg_tile.empty:
            return {}
        
        print(f"Évaluation: {len(pred_gdf)} instances prédites vs {len(rpg_tile)} parcelles RPG")
        
        # Calculer IoU par type de culture
        return self.calculate_iou_by_crop(pred_gdf, rpg_tile)

    def calculate_iou_by_crop(self, pred_gdf, rpg_tile):
        """Calcule l'IoU par type de culture avec les classes harmonisées"""
        
        results = {}
        
        # Obtenir tous les types de cultures présents
        pred_crops = set(pred_gdf['crop_type'].unique())
        rpg_crops = set(rpg_tile['classe_harmonisee'].unique())
        all_crops = pred_crops | rpg_crops
        
        print(f"Cultures à évaluer: {sorted(all_crops)}")
        
        for crop_type in all_crops:
            if crop_type is None or crop_type == 'Arrière-plan':
                continue
            
            # Géométries prédites pour cette culture
            pred_crop = pred_gdf[pred_gdf['crop_type'] == crop_type]
            pred_union = unary_union(pred_crop.geometry.tolist()) if not pred_crop.empty else None
            
            # Géométries RPG pour cette culture
            rpg_crop = rpg_tile[rpg_tile['classe_harmonisee'] == crop_type]
            rpg_union = unary_union(rpg_crop.geometry.tolist()) if not rpg_crop.empty else None
            
            # Calculer IoU
            if pred_union is None and rpg_union is None:
                continue
            elif pred_union is None:
                iou = 0.0
                intersection_area = 0.0
                pred_area = 0.0
                rpg_area = rpg_union.area
                union_area = rpg_area
            elif rpg_union is None:
                iou = 0.0
                intersection_area = 0.0
                pred_area = pred_union.area
                rpg_area = 0.0
                union_area = pred_area
            else:
                try:
                    intersection = pred_union.intersection(rpg_union)
                    union = pred_union.union(rpg_union)
                    
                    intersection_area = intersection.area if hasattr(intersection, 'area') else 0.0
                    union_area = union.area if hasattr(union, 'area') else 0.0
                    
                    iou = intersection_area / union_area if union_area > 0 else 0.0
                    pred_area = pred_union.area
                    rpg_area = rpg_union.area
                except Exception as e:
                    print(f"Erreur calcul IoU pour {crop_type}: {e}")
                    continue
            
            # Calcul précision et recall
            precision = intersection_area / pred_area if pred_area > 0 else 0.0
            recall = intersection_area / rpg_area if rpg_area > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[crop_type] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'intersection_area_m2': intersection_area,
                'union_area_m2': union_area,
                'pred_area_m2': pred_area,
                'rpg_area_m2': rpg_area,
                'pred_instances_count': len(pred_crop),
                'rpg_parcels_count': len(rpg_crop),
                'pred_instances_avg_area': pred_crop['area_m2'].mean() if not pred_crop.empty else 0,
                'rpg_parcels_avg_area': rpg_crop.geometry.area.mean() if not rpg_crop.empty else 0
            }
            
            print(f"{crop_type}: IoU={iou:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}")
        
        return results

    def analyze_instance_quality(self, pred_gdf, rpg_tile):
        """Analyse la qualité de chaque instance individuellement"""
    
        instance_analysis = []
    
        for idx, pred_instance in pred_gdf.iterrows():
        # Trouver les parcelles RPG qui intersectent cette instance
            intersecting_rpg = rpg_tile[rpg_tile.intersects(pred_instance.geometry)]
        
            if intersecting_rpg.empty:
            # Instance sans correspondance RPG
                instance_analysis.append({
                'instance_id': pred_instance['instance_id'],
                'semantic_class': pred_instance['semantic_class'],
                'crop_type': pred_instance['crop_type'],
                'status': 'no_rpg_match',
                'iou': 0.0,
                'pred_area': pred_instance['area_m2'],
                'rpg_area': 0.0,
                'confidence': pred_instance.get('confidence', None),
                'crop_type_match': False,
                'rpg_crop_type': None
                })
                continue
        
        # Calculer l'IoU avec chaque parcelle intersectante
            best_iou = 0.0
            best_match = None
        
            for rpg_idx, rpg_parcel in intersecting_rpg.iterrows():
                try:
                    intersection = pred_instance.geometry.intersection(rpg_parcel.geometry)
                    union = pred_instance.geometry.union(rpg_parcel.geometry)
                
                    iou = intersection.area / union.area if union.area > 0 else 0.0
                
                    if iou > best_iou:
                        best_iou = iou
                        best_match = rpg_parcel
                except Exception as e:
                    print(f"Erreur calcul IoU instance {pred_instance['instance_id']}: {e}")
                    continue
        
            if best_match is not None:
            # Vérifier la cohérence du type de culture
                crop_match = pred_instance['crop_type'] == best_match['classe_harmonisee']
            
                instance_analysis.append({
                'instance_id': pred_instance['instance_id'],
                'semantic_class': pred_instance['semantic_class'],
                'crop_type': pred_instance['crop_type'],
                'rpg_crop_type': best_match['classe_harmonisee'],
                'crop_type_match': crop_match,
                'iou': best_iou,
                'pred_area': pred_instance['area_m2'],
                'rpg_area': best_match.geometry.area,
                'confidence': pred_instance.get('confidence', None),
                'status': 'matched'
                })
            else:
            # Cas où il y a intersection mais échec du calcul IoU
                instance_analysis.append({
                'instance_id': pred_instance['instance_id'],
                'semantic_class': pred_instance['semantic_class'],
                'crop_type': pred_instance['crop_type'],
                'status': 'intersection_error',
                'iou': 0.0,
                'pred_area': pred_instance['area_m2'],
                'rpg_area': 0.0,
                'confidence': pred_instance.get('confidence', None),
                'crop_type_match': False,
                'rpg_crop_type': None
                })
    
        return pd.DataFrame(instance_analysis)

    def get_tile_geometry_from_geojson(self, tile_id, tiles_geojson_path):
        """Récupère la géométrie d'une tuile depuis le GeoJSON"""
        tiles_gdf = gpd.read_file(tiles_geojson_path)
    
    # Essayer différentes colonnes possibles pour l'ID
        id_columns = ['tile_id', 'id', 'ID', 'name', 'NAME']
    
        tile_geom = None
        for col in id_columns:
            if col in tiles_gdf.columns:
                matching_tiles = tiles_gdf[tiles_gdf[col].astype(str) == str(tile_id)]
                if not matching_tiles.empty:
                    tile_geom = matching_tiles.geometry.iloc[0]
                    break
    
        if tile_geom is None:
            print(f"Géométrie non trouvée pour tile_id: {tile_id}")
            return None
    
    # Assurer le bon CRS
        if tiles_gdf.crs != 'EPSG:2154':
            tile_geom = gpd.GeoSeries([tile_geom], crs=tiles_gdf.crs).to_crs('EPSG:2154').iloc[0]
    
        return tile_geom

    def generate_comprehensive_report(self, all_results, detailed_df):
        """Génère un rapport complet"""
    
    # 1. Rapport par culture (moyennes pondérées)
        crop_summary = defaultdict(lambda: {
        'ious': [], 'areas': [], 'precisions': [], 'recalls': []
        })
    
        for tile_id, tile_results in all_results.items():
            for crop_type, metrics in tile_results.items():
                crop_summary[crop_type]['ious'].append(metrics['iou'])
                crop_summary[crop_type]['areas'].append(metrics['union_area_m2'])
                crop_summary[crop_type]['precisions'].append(metrics['precision'])
                crop_summary[crop_type]['recalls'].append(metrics['recall'])
    
    # Calcul des moyennes pondérées par aire
        summary_report = {}
        for crop_type, data in crop_summary.items():
            areas = np.array(data['areas'])
            if areas.sum() > 0:
                weights = areas / areas.sum()
                summary_report[crop_type] = {
                'weighted_mean_iou': np.average(data['ious'], weights=weights),
                'weighted_mean_precision': np.average(data['precisions'], weights=weights),
                'weighted_mean_recall': np.average(data['recalls'], weights=weights),
                'total_area_m2': areas.sum(),
                'n_tiles': len(data['ious'])
                }
    
    # 2. Statistiques des instances
        instance_stats = {}
        if not detailed_df.empty:
            instance_stats = {
            'total_instances': len(detailed_df),
            'matched_instances': len(detailed_df[detailed_df['status'] == 'matched']),
            'crop_type_accuracy': detailed_df['crop_type_match'].mean() if 'crop_type_match' in detailed_df.columns else 0,
            'mean_instance_iou': detailed_df['iou'].mean(),
            'instances_per_crop': detailed_df.groupby('crop_type').size().to_dict()
            }
    
        return pd.DataFrame(summary_report).T, instance_stats

    def analyze_semantic_class_performance(self, all_results, detailed_df):
        """Analyse les performances par classe sémantique originale"""
    
        if detailed_df.empty:
            return pd.DataFrame()
    
    # Grouper par classe sémantique
        semantic_analysis = []
    
        for semantic_class in detailed_df['semantic_class'].unique():
            if semantic_class == 0:  # Ignorer l'arrière-plan
                continue
        
            class_data = detailed_df[detailed_df['semantic_class'] == semantic_class]
            crop_name = self.class_mapping.get(semantic_class, f'unknown_{semantic_class}')
            harmonized_name = self.semantic_class_to_crop_name(semantic_class)
        
            analysis = {
            'semantic_class': semantic_class,
            'class_name': crop_name,
            'harmonized_name': harmonized_name,
            'total_instances': len(class_data),
            'matched_instances': len(class_data[class_data['status'] == 'matched']),
            'match_rate': len(class_data[class_data['status'] == 'matched']) / len(class_data) if len(class_data) > 0 else 0,
            'mean_iou': class_data['iou'].mean(),
            'mean_confidence': class_data['confidence'].mean() if 'confidence' in class_data.columns and not class_data['confidence'].isna().all() else None,
            'total_pred_area_ha': class_data['pred_area'].sum() / 10000 if 'pred_area' in class_data.columns else 0,
            'crop_type_accuracy': class_data['crop_type_match'].mean() if 'crop_type_match' in class_data.columns else None
            }
        
            semantic_analysis.append(analysis)
    
        return pd.DataFrame(semantic_analysis).sort_values('semantic_class')

    def load_tile_data_from_s3(self, bucket_name, tile_key):
        """
        Charge les données d'une tuile depuis S3
    
        Args:
            bucket_name: nom du bucket S3
            tile_key: clé S3 du fichier de la tuile (ex: "predictions/tile_123.pkl")
        """
        try:
        # Construire le chemin S3 correct
            s3_path = f's3://{bucket_name}/{tile_key}'
        
            print(f"Tentative de chargement: {s3_path}")
        
        # Méthode 1: Si c'est un fichier pickle
            if tile_key.endswith('.pkl') or tile_key.endswith('.pickle'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = pickle.load(f)
                return data
        
        # Méthode 2: Si c'est un fichier numpy
            elif tile_key.endswith('.npy'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                return data
        
            elif tile_key.endswith('.npz'):
                with self.fs.open(s3_path, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                return data
        
        # Méthode 3: Format JSON
            elif tile_key.endswith('.json'):
                import json
                with self.fs.open(s3_path, 'r') as f:
                    data = json.load(f)
                return data
        
            else:
                print(f"Format de fichier non supporté: {tile_key}")
                return None
            
        except Exception as e:
            print(f"Erreur lors du chargement de {tile_key}: {e}")
            return None

    def extract_tile_id_from_key(self, tile_key):
        """Extrait l'ID de la tuile depuis la clé S3"""
    
    # Retirer le nom du bucket s'il est présent au début
        if tile_key.startswith("antoinelesauvage/"):
            tile_key = tile_key.replace("antoinelesauvage/", "", 1)
    
        filename = tile_key.split('/')[-1]
        name_without_ext = filename.split('.')[0]
    
    # Pour vos fichiers : 851_best_fold_5.npy -> 851
        import re
        match = re.search(r'(\d+)_best_fold_\d+', name_without_ext)
        if match:
            return match.group(1)
    
    # Fallback
        return name_without_ext.split('_')[0] if '_' in name_without_ext else name_without_ext

    def plot_performance_analysis(self, results_by_threshold):
        """Génère des graphiques d'analyse des performances"""
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance par seuil de confiance
        threshold_data = []
        for threshold, results in results_by_threshold.items():
            summary = results['summary']
            threshold_data.append({
            'threshold': threshold,
            'IoU': summary['weighted_mean_iou'].mean(),
            'Precision': summary['weighted_mean_precision'].mean(),
            'Recall': summary['weighted_mean_recall'].mean()
            })
    
        threshold_df = pd.DataFrame(threshold_data)
    
        axes[0,0].plot(threshold_df['threshold'], threshold_df['IoU'], 'o-', label='IoU')
        axes[0,0].plot(threshold_df['threshold'], threshold_df['Precision'], 's-', label='Precision')
        axes[0,0].plot(threshold_df['threshold'], threshold_df['Recall'], '^-', label='Recall')
        axes[0,0].set_xlabel('Seuil de confiance')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Performance vs Seuil de confiance')
        axes[0,0].legend()
        axes[0,0].grid(True)
    
    # 2. Performance par culture (meilleur seuil)
        if threshold_data:
            best_threshold = threshold_df.loc[threshold_df['IoU'].idxmax(), 'threshold']
            best_results = results_by_threshold[best_threshold]['summary']
        
            top_crops = best_results.nlargest(10, 'weighted_mean_iou')
            axes[0,1].barh(range(len(top_crops)), top_crops['weighted_mean_iou'])
            axes[0,1].set_yticks(range(len(top_crops)))
            axes[0,1].set_yticklabels(top_crops.index, fontsize=8)
            axes[0,1].set_xlabel('IoU moyen')
            axes[0,1].set_title(f'Top 10 cultures (seuil {best_threshold})')
        
        # 3. Analyse par classe sémantique
            semantic_data = results_by_threshold[best_threshold]['semantic']
            if not semantic_data.empty:
                axes[1,0].scatter(semantic_data['total_instances'], semantic_data['mean_iou'], 
                                 alpha=0.7, s=semantic_data['total_pred_area_ha']*2)
                axes[1,0].set_xlabel('Nombre d\'instances')
                axes[1,0].set_ylabel('IoU moyen')
                axes[1,0].set_title('IoU vs Nombre d\'instances\n(taille = aire prédite)')
            
            # Ajouter les labels des classes les plus importantes
                top_classes = semantic_data.nlargest(5, 'total_pred_area_ha')
                for _, row in top_classes.iterrows():
                    axes[1,0].annotate(row['class_name'][:15], 
                                  (row['total_instances'], row['mean_iou']),
                                  fontsize=8, ha='center')
    
    # 4. Distribution des aires
        axes[1,1].text(0.5, 0.5, 'Distribution des aires\n(à implémenter)', 
                   ha='center', va='center', transform=axes[1,1].transAxes)
    
        plt.tight_layout()
        plt.show()

    def save_results(self, all_results, detailed_df, output_dir='evaluation_results'):
        """Sauvegarde les résultats d'évaluation"""
        import os
        import json
    
        os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les résultats par tuile
        with open(f'{output_dir}/tile_results.json', 'w') as f:
        # Convertir les résultats en format sérialisable
            serializable_results = {}
            for tile_id, results in all_results.items():
                serializable_results[tile_id] = {
                    crop: {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                           for k, v in metrics.items()}
                    for crop, metrics in results.items()
                }
            json.dump(serializable_results, f, indent=2)
    
    # Sauvegarder l'analyse détaillée
        if not detailed_df.empty:
            detailed_df.to_csv(f'{output_dir}/detailed_instance_analysis.csv', index=False)
    
    # Générer et sauvegarder le rapport de synthèse
        summary_report, instance_stats = self.generate_comprehensive_report(all_results, detailed_df)
        summary_report.to_csv(f'{output_dir}/crop_performance_summary.csv')
    
        with open(f'{output_dir}/instance_statistics.json', 'w') as f:
            json.dump(instance_stats, f, indent=2)
    
        print(f"Résultats sauvegardés dans le dossier: {output_dir}")






# Après avoir configuré les variables d'environnement S3
evaluator = PanopticS3Evaluator(rpg_path='PARCELLES_GRAPHIQUES.shp')

# Évaluation avec différents seuils
confidence_thresholds = [0.3, 0.5, 0.7]
results_by_threshold = {}

for threshold in confidence_thresholds:
    print(f"\n=== ÉVALUATION AVEC SEUIL {threshold} ===")
    
    all_results, detailed_df = evaluator.evaluate_tiles_from_s3(
        bucket_name='antoinelesauvage',
        tiles_geojson_path='grid_indre_loire_128_lambert93.geojson',
        prefix='vergers-france/preds_128_10/',
        confidence_threshold=threshold,
        max_tiles=10  # Pour test
    )
    
    summary_report, instance_stats = evaluator.generate_comprehensive_report(all_results, detailed_df)
    semantic_analysis = evaluator.analyze_semantic_class_performance(all_results, detailed_df)
    
    results_by_threshold[threshold] = {
        'summary': summary_report,
        'instances': instance_stats,
        'semantic': semantic_analysis
    }
    
    # Sauvegarder les résultats
    evaluator.save_results(all_results, detailed_df, f'results_threshold_{threshold}')

# Visualisations
evaluator.plot_performance_analysis(results_by_threshold)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

def visualize_tile_comparison(evaluator, tile_id, instances, tile_geometry, 
                             rpg_gdf, confidence_threshold=0.5, 
                             save_path=None, figsize=(15, 10)):
    """
    Visualise la comparaison entre instances prédites et parcelles RPG pour une tuile
    
    Args:
        evaluator: instance de PanopticS3Evaluator
        tile_id: ID de la tuile
        instances: liste des instances extraites
        tile_geometry: géométrie de la tuile
        rpg_gdf: GeoDataFrame du RPG complet
        confidence_threshold: seuil de confiance
        save_path: chemin pour sauvegarder l'image
        figsize: taille de la figure
    """
    
    # Créer la figure avec 3 sous-graphiques
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Comparaison Tuile {tile_id}', fontsize=16, fontweight='bold')
    
    # Obtenir les limites de la tuile
    bounds = tile_geometry.bounds
    buffer = 50  # Buffer de 50m pour une meilleure visualisation
    
    # Filtrer le RPG pour cette zone
    rpg_tile = rpg_gdf.cx[
        bounds[0]-buffer:bounds[2]+buffer,
        bounds[1]-buffer:bounds[3]+buffer
    ].copy()
    rpg_tile = rpg_tile[rpg_tile['classe_harmonisee'].notna()]
    
    # Créer le GeoDataFrame des prédictions
    pred_gdf = evaluator.create_instances_gdf(instances, confidence_threshold)
    
    # === GRAPHIQUE 1: Parcelles RPG ===
    ax1 = axes[0]
    ax1.set_title('Parcelles RPG (Vérité terrain)', fontweight='bold')
    
    # Afficher le contour de la tuile
    tile_patch = patches.Polygon(list(tile_geometry.exterior.coords), 
                                linewidth=2, edgecolor='black', 
                                facecolor='none', linestyle='--', alpha=0.8)
    ax1.add_patch(tile_patch)
    
    # Couleurs pour les différents types de cultures RPG
    rpg_colors = plt.cm.Set3(np.linspace(0, 1, len(rpg_tile['classe_harmonisee'].unique())))
    rpg_color_map = dict(zip(rpg_tile['classe_harmonisee'].unique(), rpg_colors))
    
    # Afficher les parcelles RPG
    for idx, row in rpg_tile.iterrows():
        if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
            try:
                if row.geometry.geom_type == 'Polygon':
                    polygons = [row.geometry]
                else:
                    polygons = list(row.geometry.geoms)
                
                for poly in polygons:
                    coords = list(poly.exterior.coords)
                    patch = patches.Polygon(coords, 
                                          facecolor=rpg_color_map[row['classe_harmonisee']], 
                                          edgecolor='black', linewidth=0.5, alpha=0.7)
                    ax1.add_patch(patch)
            except:
                continue
    
    ax1.set_xlim(bounds[0]-buffer, bounds[2]+buffer)
    ax1.set_ylim(bounds[1]-buffer, bounds[3]+buffer)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Légende RPG
    rpg_legend_elements = [patches.Patch(facecolor=color, label=crop[:15]) 
                          for crop, color in rpg_color_map.items()]
    ax1.legend(handles=rpg_legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=8, title='Cultures RPG')
    
    # === GRAPHIQUE 2: Instances prédites ===
    ax2 = axes[1]
    ax2.set_title(f'Instances prédites (seuil conf. {confidence_threshold})', fontweight='bold')
    
    # Afficher le contour de la tuile
    tile_patch2 = patches.Polygon(list(tile_geometry.exterior.coords), 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='none', linestyle='--', alpha=0.8)
    ax2.add_patch(tile_patch2)
    
    if not pred_gdf.empty:
        # Couleurs pour les différents types de cultures prédites
        pred_colors = plt.cm.Set1(np.linspace(0, 1, len(pred_gdf['crop_type'].unique())))
        pred_color_map = dict(zip(pred_gdf['crop_type'].unique(), pred_colors))
        
        # Afficher les instances prédites
        for idx, row in pred_gdf.iterrows():
            if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                try:
                    if row.geometry.geom_type == 'Polygon':
                        polygons = [row.geometry]
                    else:
                        polygons = list(row.geometry.geoms)
                    
                    for poly in polygons:
                        coords = list(poly.exterior.coords)
                        # Épaisseur du contour selon la confiance
                        conf = row.get('confidence', 0.5)
                        linewidth = max(0.5, conf * 2)
                        
                        patch = patches.Polygon(coords, 
                                              facecolor=pred_color_map[row['crop_type']], 
                                              edgecolor='red', linewidth=linewidth, alpha=0.6)
                        ax2.add_patch(patch)
                        
                        # Ajouter l'ID de l'instance
                        centroid = poly.centroid
                        ax2.text(centroid.x, centroid.y, str(row['instance_id']), 
                                ha='center', va='center', fontsize=8, fontweight='bold')
                except:
                    continue
        
        # Légende prédictions
        pred_legend_elements = [patches.Patch(facecolor=color, label=crop[:15]) 
                              for crop, color in pred_color_map.items()]
        ax2.legend(handles=pred_legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=8, title='Cultures prédites')
    else:
        ax2.text(0.5, 0.5, 'Aucune instance\nau-dessus du seuil', 
                transform=ax2.transAxes, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    ax2.set_xlim(bounds[0]-buffer, bounds[2]+buffer)
    ax2.set_ylim(bounds[1]-buffer, bounds[3]+buffer)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # === GRAPHIQUE 3: Superposition avec analyse ===
    ax3 = axes[2]
    ax3.set_title('Superposition et analyse des correspondances', fontweight='bold')
    
    # Afficher le contour de la tuile
    tile_patch3 = patches.Polygon(list(tile_geometry.exterior.coords), 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='none', linestyle='--', alpha=0.8)
    ax3.add_patch(tile_patch3)
    
    # D'abord les parcelles RPG (en transparence)
    for idx, row in rpg_tile.iterrows():
        if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
            try:
                if row.geometry.geom_type == 'Polygon':
                    polygons = [row.geometry]
                else:
                    polygons = list(row.geometry.geoms)
                
                for poly in polygons:
                    coords = list(poly.exterior.coords)
                    patch = patches.Polygon(coords, 
                                          facecolor='lightgray', 
                                          edgecolor='gray', linewidth=1, alpha=0.3)
                    ax3.add_patch(patch)
            except:
                continue
    
    # Puis les instances prédites avec code couleur selon la qualité
    if not pred_gdf.empty:
        # Analyser la qualité des instances
        instance_analysis = evaluator.analyze_instance_quality(pred_gdf, rpg_tile)
        
        for idx, row in pred_gdf.iterrows():
            if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                try:
                    # Trouver l'analyse correspondante
                    analysis = instance_analysis[
                        instance_analysis['instance_id'] == row['instance_id']
                    ]
                    
                    if not analysis.empty:
                        analysis_row = analysis.iloc[0]
                        
                        # Code couleur selon la qualité
                        if analysis_row['status'] == 'matched':
                            if analysis_row['crop_type_match']:
                                if analysis_row['iou'] > 0.7:
                                    color = 'green'  # Très bon match
                                elif analysis_row['iou'] > 0.5:
                                    color = 'lightgreen'  # Bon match
                                else:
                                    color = 'yellow'  # Match moyen
                            else:
                                color = 'orange'  # Mauvais type de culture
                        else:
                            color = 'red'  # Pas de correspondance
                    else:
                        color = 'purple'  # Erreur d'analyse
                    
                    if row.geometry.geom_type == 'Polygon':
                        polygons = [row.geometry]
                    else:
                        polygons = list(row.geometry.geoms)
                    
                    for poly in polygons:
                        coords = list(poly.exterior.coords)
                        patch = patches.Polygon(coords, 
                                              facecolor=color, 
                                              edgecolor='black', linewidth=1.5, alpha=0.7)
                        ax3.add_patch(patch)
                        
                        # Ajouter informations textuelles
                        centroid = poly.centroid
                        if not analysis.empty:
                            info_text = f"ID:{row['instance_id']}\nIoU:{analysis_row['iou']:.2f}"
                            ax3.text(centroid.x, centroid.y, info_text, 
                                    ha='center', va='center', fontsize=7, 
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                except:
                    continue
    
    ax3.set_xlim(bounds[0]-buffer, bounds[2]+buffer)
    ax3.set_ylim(bounds[1]-buffer, bounds[3]+buffer)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Légende pour la qualité des correspondances
    quality_legend = [
        patches.Patch(facecolor='green', label='Excellent (IoU>0.7, bon type)'),
        patches.Patch(facecolor='lightgreen', label='Bon (IoU>0.5, bon type)'),
        patches.Patch(facecolor='yellow', label='Moyen (IoU<0.5, bon type)'),
        patches.Patch(facecolor='orange', label='Mauvais type de culture'),
        patches.Patch(facecolor='red', label='Pas de correspondance RPG'),
        patches.Patch(facecolor='lightgray', label='Parcelles RPG')
    ]
    ax3.legend(handles=quality_legend, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=8, title='Qualité des correspondances')
    
    # Statistiques globales
    if not pred_gdf.empty and not instance_analysis.empty:
        stats_text = f"""Statistiques tuile {tile_id}:
• Instances prédites: {len(pred_gdf)}
• Parcelles RPG: {len(rpg_tile)}
• Correspondances trouvées: {len(instance_analysis[instance_analysis['status'] == 'matched'])}
• Bons types de culture: {len(instance_analysis[instance_analysis['crop_type_match'] == True])}
• IoU moyen: {instance_analysis['iou'].mean():.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.show()
    
    return fig, axes

def visualize_multiple_tiles(evaluator, bucket_name, tiles_geojson_path, 
                           prefix="", confidence_threshold=0.5, 
                           n_tiles=5, output_dir='tile_visualizations'):
    """
    Visualise plusieurs tuiles aléatoires
    
    Args:
        evaluator: instance de PanopticS3Evaluator
        bucket_name: nom du bucket S3
        tiles_geojson_path: chemin vers le GeoJSON des tuiles
        prefix: préfixe S3
        confidence_threshold: seuil de confiance
        n_tiles: nombre de tuiles à visualiser
        output_dir: dossier de sortie
    """
    import os
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister les tuiles disponibles
    tile_files = evaluator.list_tiles_in_s3(bucket_name, prefix)
    
    # Sélectionner aléatoirement n_tiles
    selected_tiles = random.sample(tile_files, min(n_tiles, len(tile_files)))
    
    print(f"Visualisation de {len(selected_tiles)} tuiles...")
    
    for i, tile_key in enumerate(selected_tiles):
        print(f"\n--- Visualisation tuile {i+1}/{len(selected_tiles)}: {tile_key} ---")
        
        # Extraire l'ID et charger les données
        tile_id = evaluator.extract_tile_id_from_key(tile_key)
        tile_data = evaluator.load_tile_data_from_s3(bucket_name, tile_key)
        
        if tile_data is None:
            print(f"Impossible de charger {tile_key}")
            continue
        
        # Récupérer la géométrie
        tile_geometry = evaluator.get_tile_geometry_from_geojson(tile_id, tiles_geojson_path)
        if tile_geometry is None:
            print(f"Géométrie non trouvée pour {tile_id}")
            continue
        
        # Extraire les instances
        try:
            instances = evaluator.extract_instances_from_tile_data(
                tile_data, tile_id, tile_geometry, 'EPSG:2154'
            )
        except Exception as e:
            print(f"Erreur extraction instances: {e}")
            continue
        
        if not instances:
            print(f"Aucune instance pour {tile_id}")
            continue
        
        # Créer la visualisation
        save_path = f"{output_dir}/tile_{tile_id}_comparison.png"
        
        try:
            visualize_tile_comparison(
                evaluator, tile_id, instances, tile_geometry, 
                evaluator.rpg, confidence_threshold, save_path
            )
        except Exception as e:
            print(f"Erreur visualisation {tile_id}: {e}")
            continue
    
    print(f"\nVisualisations sauvegardées dans: {output_dir}")

# Fonction utilitaire pour visualiser une tuile spécifique
def visualize_specific_tile(evaluator, bucket_name, tile_key, tiles_geojson_path, 
                           confidence_threshold=0.5, save_path=None):
    """Visualise une tuile spécifique"""
    
    print(f"Visualisation de la tuile: {tile_key}")
    
    # Extraire l'ID et charger les données
    tile_id = evaluator.extract_tile_id_from_key(tile_key)
    tile_data = evaluator.load_tile_data_from_s3(bucket_name, tile_key)
    
    if tile_data is None:
        print(f"Impossible de charger {tile_key}")
        return None
    
    # Récupérer la géométrie
    tile_geometry = evaluator.get_tile_geometry_from_geojson(tile_id, tiles_geojson_path)
    if tile_geometry is None:
        print(f"Géométrie non trouvée pour {tile_id}")
        return None
    
    # Extraire les instances
    instances = evaluator.extract_instances_from_tile_data(
        tile_data, tile_id, tile_geometry, 'EPSG:2154'
    )
    
    if not instances:
        print(f"Aucune instance pour {tile_id}")
        return None
    
    # Créer la visualisation
    return visualize_tile_comparison(
        evaluator, tile_id, instances, tile_geometry, 
        evaluator.rpg, confidence_threshold, save_path
    )

# Visualiser une tuile spécifique
tile_key = "vergers-france/preds_128_10/851_best_fold_5.npy"
fig, axes = visualize_specific_tile(
    evaluator, 
    bucket_name='antoinelesauvage',
    tile_key=tile_key,
    tiles_geojson_path='grid_indre_loire_128_lambert93.geojson',
    confidence_threshold=0.5,
    save_path='tile_851_comparison.png'
)

# Visualiser plusieurs tuiles aléatoires
visualize_multiple_tiles(
    evaluator,
    bucket_name='antoinelesauvage',
    tiles_geojson_path='grid_indre_loire_128_lambert93.geojson',
    prefix='vergers-france/preds_128_10/',
    confidence_threshold=0.5,
    n_tiles=5,
    output_dir='tile_visualizations'
)