from pathlib import Path
import numpy as np
import pycolmap

from utils import get_poses_images ,get_focal

def colmap_match(image_dir ,output_path):

    output_path.mkdir(exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)


def generate_meta_data(output_path):

    reconstruction = pycolmap.Reconstruction(output_path)

    images ,poses = get_poses_images(reconstruction)

    poses = np.array(poses)
    images = np.array(images)

    H, W = images.shape[1:3]

    focal = get_focal(reconstruction)

    return images ,poses ,H ,W ,focal