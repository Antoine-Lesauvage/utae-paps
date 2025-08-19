import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from affine import Affine
from rasterio.features import shapes
import cv2
import os
import re
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm
import gc

def setup_s3_client():
    """
    Configure le client S3 pour Onyxia
    """
    try:
        s3_client = boto3.client("s3",endpoint_url = 'https://'+'minio.lab.sspcloud.fr',
                  aws_access_key_id= '00AYNA8AVQBH6DZSIUDP', 
                  aws_secret_access_key= 'T7aXS+eOSs9tYDFlvAr6LcHuNEf1yId7phfaNRT7', 
                  aws_session_token = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiIwMEFZTkE4QVZRQkg2RFpTSVVEUCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU1MTYxNjUzLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU1NzY3MjA3LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NTE2MjQwNywiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDpkMWM0YTU3ZS0yMjNiLWU3YjYtYTg0Ni0xODZiZDQ0MTcxMGQiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiJmZjY1ZDg0NS02N2M4LTQxYjUtOTNmOS1lNzQ0NDdlYjVlMmIiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.L4I0HfqMXHWFyk864CTo9_tIDZbbx2scbVzJ70Jncdokv9IrmaaXnQIks_Mfg4NLyMUnVeNYNvRGrMBITLTtjQ')
        return s3_client
    except Exception as e:
        print(f"Erreur lors de la configuration S3: {e}")
        return None

def list_s3_npy_files(s3_client, bucket_name, prefix):
    """
    Liste tous les fichiers .npy dans le bucket S3 avec le pattern attendu
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        npy_files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.npy') and '_best_fold_' in key:
                        npy_files.append(key)
        
        print(f"Trouvé {len(npy_files)} fichiers .npy avec le pattern *_best_fold_*.npy sur S3")
        return npy_files
    except Exception as e:
        print(f"Erreur lors de la liste des fichiers S3: {e}")
        return []

def load_npy_from_s3(s3_client, bucket_name, key):
    """
    Charge un fichier .npy depuis S3 avec retry
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            data = response['Body'].read()
            
            with io.BytesIO(data) as bio:
                array_data = np.load(bio, allow_pickle=True)
                
            return array_data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff exponentiel
                continue
            else:
                print(f"Erreur définitive lors du chargement de {key}: {e}")
                return None

def process_single_tile(args):
    """
    Traite une seule tuile - fonction pour le multiprocessing
    """
    (s3_key, bucket_name, tile_id, tile_geometry, grid_crs) = args
    
    try:
        # Configurer un nouveau client S3 pour ce processus (nécessaire pour multiprocessing)
        s3_client = setup_s3_client()
        if s3_client is None:
            return None
        
        # Charger les données depuis S3
        data = load_npy_from_s3(s3_client, bucket_name, s3_key)
        
        if data is None:
            return None
        
        # Extraire les instances
        gdf = extract_instances_from_tile_data(
            data,
            tile_id, 
            tile_geometry, 
            grid_crs
        )
        
        # Nettoyer la mémoire
        del data
        gc.collect()
        
        return gdf
        
    except Exception as e:
        print(f"Erreur lors du traitement de {s3_key}: {e}")
        return None

def batch_process_tiles(tile_batch, bucket_name, grid, grid_crs, id_column):
    """
    Traite un batch de tuiles de manière séquentielle
    """
    results = []
    
    for s3_key in tile_batch:
        filename = Path(s3_key).name
        
        # Trouver l'ID de la tuile
        tile_id = find_tile_id_from_filename(filename, grid)
        
        if tile_id is None:
            continue
        
        # Trouver la géométrie correspondante dans la grille
        tile_row = None
        
        try:
            tile_row = grid[pd.to_numeric(grid[id_column], errors='coerce') == tile_id]
        except:
            pass
        
        if tile_row is None or len(tile_row) == 0:
            tile_row = grid[grid[id_column] == str(tile_id)]
        
        if len(tile_row) == 0:
            continue
        
        tile_geometry = tile_row.geometry.iloc[0]
        
        # Traiter la tuile
        try:
            s3_client = setup_s3_client()
            if s3_client is None:
                continue
                
            data = load_npy_from_s3(s3_client, bucket_name, s3_key)
            
            if data is None:
                continue
            
            gdf = extract_instances_from_tile_data(
                data,
                tile_id, 
                tile_geometry, 
                grid_crs
            )
            
            if gdf is not None and len(gdf) > 0:
                results.append(gdf)
            
            # Nettoyer la mémoire
            del data
            gc.collect()
            
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {e}")
            continue
    
    return results

def load_tile_grid(grid_geojson_path):
    """
    Charge la grille des tuiles et la reprojette en coordonnées métriques
    """
    if grid_geojson_path.startswith('s3://'):
        parts = grid_geojson_path.replace('s3://', '').split('/', 1)
        bucket_name = parts[0]
        key = parts[1]
        
        s3_client = setup_s3_client()
        if s3_client is None:
            raise Exception("Impossible de configurer le client S3")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp_file:
                s3_client.download_file(bucket_name, key, tmp_file.name)
                grid = gpd.read_file(tmp_file.name)
                os.unlink(tmp_file.name)
        except Exception as e:
            print(f"Erreur lors du chargement de la grille depuis S3: {e}")
            raise
    else:
        grid = gpd.read_file(grid_geojson_path)
    
    print(f"Grille chargée: {len(grid)} tuiles")
    print(f"CRS original: {grid.crs}")
    
    if grid.crs != 'EPSG:2154':
        print("Reprojection vers Lambert 93 (EPSG:2154)")
        grid = grid.to_crs('EPSG:2154')
        print(f"CRS après reprojection: {grid.crs}")
    
    return grid

# Garder toutes les autres fonctions existantes (get_tile_transform, find_tile_id_from_filename, 
# calculate_box_iou, extract_instances_from_tile_data, mask_to_polygons_georef, 
# mask_to_polygons_opencv) inchangées...

def get_tile_transform(tile_geometry, image_shape):
    """
    Calcule la transformation affine pour passer des coordonnées pixel aux coordonnées géographiques
    """
    bounds = tile_geometry.bounds
    
    width_geo = bounds[2] - bounds[0]
    height_geo = bounds[3] - bounds[1]
    
    pixel_width = width_geo / image_shape[1]
    pixel_height = height_geo / image_shape[0]
    
    transform = Affine(
        pixel_width, 0.0, bounds[0],
        0.0, -pixel_height, bounds[3]
    )
    
    return transform

def find_tile_id_from_filename(filename, grid):
    """
    Trouve l'ID de la tuile à partir du nom de fichier
    """
    basename = Path(filename).stem
    pattern = r'^(\d+)_best_fold_'
    match = re.match(pattern, basename)
    
    if match:
        tile_id = int(match.group(1))
        return tile_id
    else:
        return None

def calculate_box_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes englobantes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def extract_instances_from_tile_data(data, tile_id, tile_geometry, grid_crs):
    """
    Extraction avec géoréférencement correct en coordonnées métriques
    """
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
                    iou = calculate_box_iou(instance_box, pred_box)
                    
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
    transform = get_tile_transform(tile_geometry, pano_instance.shape)
    
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
            polygons = mask_to_polygons_georef(mask, transform)
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
    
    if results:
        gdf = gpd.GeoDataFrame(results, crs=grid_crs)
        return gdf
    else:
        return None

def mask_to_polygons_georef(mask, transform):
    """
    Convertit un masque binaire en polygones géoréférencés
    """
    polygons = []
    
    try:
        for shape, value in shapes(mask, mask=mask > 0, transform=transform):
            if value == 1:
                coords = shape['coordinates']
                if len(coords) > 0:
                    exterior = coords[0]
                    holes = coords[1:] if len(coords) > 1 else None
                    
                    try:
                        poly = Polygon(exterior, holes)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
                    except:
                        try:
                            poly = Polygon(exterior).buffer(0)
                            if poly.is_valid and poly.area > 0:
                                polygons.append(poly)
                        except:
                            continue
    except Exception as e:
        polygons = mask_to_polygons_opencv(mask, transform)
    
    return polygons

def mask_to_polygons_opencv(mask, transform):
    """
    Méthode alternative avec OpenCV
    """
    polygons = []
    
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 3:
                epsilon = 0.5
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(simplified) >= 3:
                    coords = []
                    for point in simplified:
                        x, y = point[0]
                        geo_x, geo_y = transform * (x, y)
                        coords.append((geo_x, geo_y))
                    
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
                    except:
                        continue
    except Exception as e:
        pass
    
    return polygons

def process_all_tiles_s3_parallel(bucket_name, s3_prefix, grid_geojson_path, output_parquet_path, 
                                   n_workers=None, batch_size=100):
    """
    Version parallélisée pour traiter tous les fichiers .npy d'un bucket S3
    
    Parameters:
    - n_workers: nombre de processus (par défaut: nombre de CPUs)
    - batch_size: nombre de tuiles par batch pour équilibrer charge/mémoire
    """
    
    if n_workers is None:
        n_workers = min(cpu_count(), 16)  # Limiter à 16 pour éviter surcharge
    
    print(f"Utilisation de {n_workers} processus parallèles")
    
    # Configurer S3
    s3_client = setup_s3_client()
    if s3_client is None:
        print("Impossible de configurer le client S3")
        return None
    
    # Charger la grille
    grid = load_tile_grid(grid_geojson_path)
    
    # Identifier la colonne d'ID
    id_column = None
    possible_id_columns = ['id', 'tile_id', 'ID', 'index', 'fid']
    
    for col in possible_id_columns:
        if col in grid.columns:
            id_column = col
            print(f"Utilisation de la colonne '{col}' comme ID de tuile")
            break
    
    if id_column is None:
        print("Aucune colonne d'ID trouvée, utilisation de l'index")
        grid['tile_id'] = grid.index
        id_column = 'tile_id'
    
    # Lister tous les fichiers .npy
    print("Listage des fichiers sur S3...")
    npy_files = list_s3_npy_files(s3_client, bucket_name, s3_prefix)
    
    if len(npy_files) == 0:
        print("Aucun fichier trouvé!")
        return None
    
    print(f"Traitement de {len(npy_files)} fichiers par batches de {batch_size}")
    
    # Diviser en batches
    tile_batches = [npy_files[i:i + batch_size] for i in range(0, len(npy_files), batch_size)]
    
    print(f"Création de {len(tile_batches)} batches")
    
    all_results = []
    processed_tiles = 0
    failed_tiles = 0
    
    # Fonction partielle pour le multiprocessing
    process_batch_func = partial(batch_process_tiles, 
                                bucket_name=bucket_name, 
                                grid=grid, 
                                grid_crs=grid.crs, 
                                id_column=id_column)
    
    # Traitement parallèle par batches
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        # Utiliser tqdm pour la barre de progression
        batch_results = list(tqdm(
            pool.imap(process_batch_func, tile_batches),
            total=len(tile_batches),
            desc="Traitement des batches"
        ))
    
    # Consolider les résultats
    for batch_result in batch_results:
        if batch_result:
            all_results.extend(batch_result)
            processed_tiles += sum(len(gdf) for gdf in batch_result if gdf is not None)
    
    processing_time = time.time() - start_time
    
    # Combiner tous les résultats
    if all_results:
        print("Consolidation des résultats...")
        final_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True))
        
        # Ajouter des métadonnées
        final_gdf['unique_id'] = final_gdf.index
        final_gdf['area_ha'] = final_gdf['area_m2'] / 10000
        final_gdf['extraction_date'] = pd.Timestamp.now()
        
        # Exporter
        print("Sauvegarde en cours...")
        final_gdf.to_parquet(output_parquet_path)
        
        print(f"\n=== RÉSUMÉ ===")
        print(f"Temps de traitement: {processing_time/60:.1f} minutes")
        print(f"Vitesse: {len(npy_files)/processing_time:.1f} tuiles/seconde")
        print(f"Total instances extraites: {len(final_gdf)}")
        print(f"Nombre de tuiles avec instances: {final_gdf['tile_id'].nunique()}")
        print(f"Classes trouvées: {sorted(final_gdf['semantic_class'].unique())}")
        print(f"Surface totale: {final_gdf['area_ha'].sum():.2f} ha")
        print(f"Fichier sauvegardé: {output_parquet_path}")
        
        return final_gdf
    else:
        print("Aucune instance extraite")
        return None

def add_class_names(gdf, class_mapping=None):
    """
    Ajoute les noms des classes de cultures
    """
    if class_mapping is None:
        class_mapping = {
            0: 'Arrière-plan',
            1: 'Prairie', 
            2: 'Blé tendre hiver',
            3: 'Maïs',
            4: 'Orge hiver',
            5: 'Colza hiver',
            6: 'Orge printemps',
            7: 'Tournesol',
            8: 'Vigne',
            9: 'Betterave',
            10: 'Triticale hiver',
            11: 'Blé dur hiver',
            12: 'Fruits/légumes/fleurs',
            13: 'Pommes de terre',
            14: 'Fourrage légumineux',
            15: 'Soja',
            16: 'Verger',
            17: 'Céréale mélangée',
            18: 'Sorgho',
            19: 'Étiquette vide'
        }
    
    gdf['semantic_class'] = gdf['semantic_class'].astype(int)
    gdf['class_name'] = gdf['semantic_class'].map(class_mapping)
    gdf['class_name'] = gdf['class_name'].fillna('class_' + gdf['semantic_class'].astype(str))
    
    return gdf

# Exemple d'utilisation parallélisée
if __name__ == "__main__":
    bucket_name = "antoinelesauvage"
    s3_prefix = "vergers-france/preds/"
    grid_geojson = "grid_indre_loire_128.geojson"
    output_parquet = "agricultural_instances_indre_loire_parallel.parquet"
    
    print("=== TRAITEMENT PARALLÉLISÉ AVEC DONNÉES S3 ===")
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {s3_prefix}")
    
    # Traitement parallélisé avec paramètres optimisés pour 10k+ tuiles
    gdf = process_all_tiles_s3_parallel(
        bucket_name=bucket_name,
        s3_prefix=s3_prefix,
        grid_geojson_path=grid_geojson,
        output_parquet_path=output_parquet,
        n_workers=12,  # Ajustez selon vos ressources
        batch_size=50   # Batches plus petits pour 10k+ tuiles
    )
    
    if gdf is not None:
        # Ajouter les noms des classes
        gdf = add_class_names(gdf)
        
        # Statistiques par classe
        stats = gdf.groupby('class_name').agg({
            'unique_id': 'count',
            'area_ha': ['sum', 'mean', 'std'],
            'confidence': 'mean' if 'confidence' in gdf.columns else lambda x: None
        }).round(4)
        
        print("\n=== STATISTIQUES PAR CLASSE ===")
        print(stats)
        
        # Sauvegarder avec les noms de classes
        output_with_names = output_parquet.replace('.parquet', '_with_names.parquet')
        gdf.to_parquet(output_with_names)
        print(f"Fichier avec noms de classes sauvegardé: {output_with_names}")
        
        # Optionnel: export shapefile
        #try:
        #    gdf.to_file(output_parquet.replace('.parquet', '.shp'))
        #    print("Fichier shapefile également créé")
        #except Exception as e:
        #    print(f"Impossible de créer le shapefile: {e}")
        
        print(f"\n=== RÉSUMÉ FINAL ===")
        print(f"Total polygones: {len(gdf)}")
        print(f"Surface totale: {gdf['area_ha'].sum():.2f} ha")
        print(f"Surface moyenne par parcelle: {gdf['area_ha'].mean():.4f} ha")