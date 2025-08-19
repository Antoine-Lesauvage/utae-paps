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
                  aws_access_key_id= '15P7O0ZZKX002DJ3MCMH', 
                  aws_secret_access_key= 'HVF1EPdWT+6y5wOCjKLz3NKFrDxNs5AsiluDf8HI', 
                  aws_session_token = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiIxNVA3TzBaWktYMDAyREozTUNNSCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzU1NTAxODc0LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6ImFudG9pbmUubGVzYXV2YWdlQGFncmljdWx0dXJlLmdvdXYuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzU2MTE0NjE2LCJmYW1pbHlfbmFtZSI6IkxFU0FVVkFHRSIsImdpdmVuX25hbWUiOiJBbnRvaW5lIiwiZ3JvdXBzIjpbIlVTRVJfT05ZWElBIl0sImlhdCI6MTc1NTUwOTgxNiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDphZmYzNTk3Zi0wMDRlLTU5ZDEtZGQ1Mi01OTQ1MjBkMjBlNWEiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJBbnRvaW5lIExFU0FVVkFHRSIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbnRvaW5lbGVzYXV2YWdlIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiIzNGJiOGE3Ni05YWM4LTQ3NDItYWM3Zi05OTFhZGI3NDUxYmEiLCJzdWIiOiJlOWY4NjIyYy01MzQ2LTRjOGUtOWZmYi1jY2MwMzZjY2ZjZjciLCJ0eXAiOiJCZWFyZXIifQ.0xipTWD3t0hijByW7WGHMJEaFlut_W8H1sIJzjbwnrNo2ox67Xdw_h4IaumSCwUFtBF6XO2JFVd9vpLN7M4JFg')
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
                    if key.endswith('.npy'):  # Modifié pour accepter tous les .npy
                        npy_files.append(key)
        
        print(f"Trouvé {len(npy_files)} fichiers .npy sur S3")
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
        
        # Extraire les instances sémantiques
        gdf = extract_semantic_instances_from_tile_data(
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
            
            gdf = extract_semantic_instances_from_tile_data(
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
    Modifié pour gérer différents patterns de noms de fichiers
    """
    basename = Path(filename).stem
    
    # Essayer différents patterns
    patterns = [
        r'^(\d+)_best_fold_',  # Pattern original
        r'^(\d+)_semantic',    # Pattern sémantique
        r'^(\d+)\.npy$',       # Simple numéro
        r'tile_(\d+)',         # tile_ID
        r'^(\d+)_'             # ID suivi d'underscore
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            tile_id = int(match.group(1))
            return tile_id
    
    # Si aucun pattern ne marche, essayer d'extraire le premier nombre
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return int(numbers[0])
    
    return None

def extract_semantic_instances_from_tile_data(data, tile_id, tile_geometry, grid_crs):
    """
    Extraction spécialisée pour les données sémantiques UTAE
    Données attendues: ['semantic_pred', 'confidence_map', 'raw_output']
    """
    if data is None:
        return None
    
    # Si c'est un dictionnaire
    if isinstance(data, dict):
        semantic_pred = data.get('semantic_pred')
        confidence_map = data.get('confidence_map')
        raw_output = data.get('raw_output')
    else:
        try:
            data_dict = data.item()
            semantic_pred = data_dict.get('semantic_pred')
            confidence_map = data_dict.get('confidence_map')
            raw_output = data_dict.get('raw_output')
        except:
            # Si pas de structure dict, essayer d'interpréter directement
            if hasattr(data, 'shape') and len(data.shape) >= 2:
                semantic_pred = data
                confidence_map = None
                raw_output = None
            else:
                return None
    
    if semantic_pred is None:
        return None
    
    # Traiter les dimensions de semantic_pred
    if len(semantic_pred.shape) == 4:
        if semantic_pred.shape[0] == 1:
            semantic_pred = semantic_pred.squeeze(0)
        # Si c'est des probabilités, prendre l'argmax
        semantic_pred = np.argmax(semantic_pred, axis=0)
    elif len(semantic_pred.shape) == 3 and semantic_pred.shape[0] == 1:
        semantic_pred = semantic_pred.squeeze(0)
    
    # Traiter confidence_map si disponible
    if confidence_map is not None:
        if len(confidence_map.shape) == 3 and confidence_map.shape[0] == 1:
            confidence_map = confidence_map.squeeze(0)
        elif len(confidence_map.shape) == 4:
            if confidence_map.shape[0] == 1:
                confidence_map = confidence_map.squeeze(0)
            # Pour une carte de confiance multi-classe, prendre le max
            confidence_map = np.max(confidence_map, axis=0)
    
    # Calculer la transformation géographique
    transform = get_tile_transform(tile_geometry, semantic_pred.shape)
    
    # Traiter chaque classe sémantique
    results = []
    unique_classes = np.unique(semantic_pred)
    
    # Exclure la classe de fond (0) si elle existe
    unique_classes = unique_classes[unique_classes > 0]
    
    for semantic_class in unique_classes:
        # Créer le masque pour cette classe
        class_mask = (semantic_pred == semantic_class).astype(np.uint8)
        
        # Calculer la confiance moyenne pour cette classe si disponible
        avg_confidence = None
        if confidence_map is not None:
            class_pixels = semantic_pred == semantic_class
            class_confidences = confidence_map[class_pixels]
            if len(class_confidences) > 0:
                avg_confidence = float(np.mean(class_confidences))
        
        # Convertir en polygones avec séparation des composantes connexes
        try:
            polygons = mask_to_polygons_with_components(class_mask, transform)
        except Exception as e:
            print(f"Erreur lors de la vectorisation pour la classe {semantic_class}: {e}")
            continue
        
        if len(polygons) == 0:
            continue
        
        # Créer une entrée pour chaque polygone (composante connexe)
        for i, poly in enumerate(polygons):
            if poly.area > 0:
                result = {
                    'tile_id': tile_id,
                    'semantic_class': int(semantic_class),
                    'component_id': i,  # ID de la composante connexe
                    'geometry': poly,
                    'area_pixels': np.sum(class_mask),  # Aire totale de la classe
                    'area_m2': poly.area,  # Aire de cette composante
                    'perimeter_m': poly.length,
                    'centroid_x': poly.centroid.x,
                    'centroid_y': poly.centroid.y
                }
                
                if avg_confidence is not None:
                    result['confidence'] = avg_confidence
                
                results.append(result)
    
    if results:
        gdf = gpd.GeoDataFrame(results, crs=grid_crs)
        return gdf
    else:
        return None

def mask_to_polygons_with_components(mask, transform):
    """
    Convertit un masque binaire en polygones géoréférencés
    en séparant les composantes connexes
    """
    polygons = []
    
    try:
        # Méthode avec rasterio.features.shapes qui gère automatiquement les composantes
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
        # Méthode alternative avec OpenCV
        polygons = mask_to_polygons_opencv_components(mask, transform)
    
    return polygons

def mask_to_polygons_opencv_components(mask, transform):
    """
    Méthode alternative avec OpenCV pour gérer les composantes connexes
    """
    polygons = []
    
    try:
        # Trouver toutes les composantes connexes
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
    Version parallélisée pour traiter tous les fichiers .npy sémantiques d'un bucket S3
    
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
        if 'confidence' in final_gdf.columns:
            print(f"Confiance moyenne: {final_gdf['confidence'].mean():.3f}")
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

# Exemple d'utilisation pour les données sémantiques
if __name__ == "__main__":
    bucket_name = "antoinelesauvage"
    s3_prefix = "vergers-france/preds_sem/"  # Modifié pour les données sémantiques
    grid_geojson = "grid_indre_loire_128.geojson"
    output_parquet = "semantic_instances_indre_loire_parallel.parquet"
    
    print("=== TRAITEMENT PARALLÉLISÉ SÉMANTIQUE UTAE ===")
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {s3_prefix}")
    
    # Traitement parallélisé optimisé pour les données sémantiques
    gdf = process_all_tiles_s3_parallel(
        bucket_name=bucket_name,
        s3_prefix=s3_prefix,
        grid_geojson_path=grid_geojson,
        output_parquet_path=output_parquet,
        n_workers=12,  # Ajustez selon vos ressources
        batch_size=50   # Batches plus petits pour la stabilité
    )
    
    if gdf is not None:
        # Ajouter les noms des classes
        gdf = add_class_names(gdf)
        
        # Statistiques par classe
        stats_cols = ['unique_id', 'area_ha']
        if 'confidence' in gdf.columns:
            stats_cols.append('confidence')
        
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
        
        print(f"\n=== RÉSUMÉ FINAL ===")
        print(f"Total polygones: {len(gdf)}")
        print(f"Surface totale: {gdf['area_ha'].sum():.2f} ha")
        print(f"Surface moyenne par polygone: {gdf['area_ha'].mean():.4f} ha")
        print(f"Nombre de composantes par classe:")
        for class_name in gdf['class_name'].unique():
            count = len(gdf[gdf['class_name'] == class_name])
            print(f"  {class_name}: {count} polygones")