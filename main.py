from pathlib import Path

from colmap import colmap_match ,generate_meta_data
from nerf import *
from utils import generate_best_3d_obj

image_dir = Path('images')
output_path = Path('output') 

colmap_match(image_dir ,output_path)
images ,poses ,H ,W ,focal = generate_meta_data(output_path)

testimg ,testpose = images[-1] ,poses[-1]
images ,poses = images[:-1] ,poses[:-1]

L_embed = 6
embed_fn = posenc

model = train_model(images ,poses ,H ,W ,focal ,testimg ,testpose ,L_embed ,embed_fn)

# Usage example
output_path = 'output_mesh.obj'
generate_best_3d_obj(model, embed_fn, output_path)