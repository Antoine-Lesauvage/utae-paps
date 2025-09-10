import numpy as np
import s3fs
from torch.utils.data import Dataset

class S3CubeDataset(Dataset):
    def __init__(self, s3_prefix, tile_ids, transform=None):
        """
        s3_prefix : chemin S3 vers les .npy (ex : s3://bucket/data_2018)
        tile_ids : liste de noms de tuile (sans extension)
        """
        self.s3_prefix = s3_prefix.rstrip("/")
        self.tile_ids = tile_ids
        self.transform = transform
        self.fs = s3fs.S3FileSystem()

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        fpath = f"{self.s3_prefix}/{tile_id}.npy"

        with self.fs.open(fpath, 'rb') as f:
            data = np.load(f)  # data shape: (T, C, H, W)

        if data.ndim != 4:
            raise ValueError(f"Données mal formées dans {fpath} : shape {data.shape}")

        if self.transform:
            data = self.transform(data)

        return {
            "data": data,       # (T, C, H, W)
            "tile_id": tile_id
        }