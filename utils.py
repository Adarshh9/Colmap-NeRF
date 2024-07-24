from moviepy.editor import VideoFileClip
from skimage.transform import resize
from PIL import Image as PILImage
from pathlib import Path
import numpy as np
import tensorflow as tf
import trimesh
import skimage.measure


# Function to sample 3D points
def sample_3d_points(resolution, bounding_box):
    x = np.linspace(bounding_box[0][0], bounding_box[0][1], resolution)
    y = np.linspace(bounding_box[1][0], bounding_box[1][1], resolution)
    z = np.linspace(bounding_box[2][0], bounding_box[2][1], resolution)
    grid = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack(grid, axis=-1)
    return points


# Function to query the NeRF model
def query_nerf(model, points, embed_fn):
    points_flat = points.reshape(-1, 3)
    points_flat = embed_fn(points_flat)
    raw = model(points_flat)
    raw = tf.reshape(raw, tuple(points.shape[:-1]) + (4,))
    return raw


# Function to convert density to a mesh using marching cubes
def density_to_mesh(density, threshold):
    verts, faces, _, _ = skimage.measure.marching_cubes(density, threshold)
    return verts, faces


# Function to save mesh to .obj file
def save_mesh_to_obj(verts, faces, filename):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)


# Function to evaluate the quality of a mesh (dummy implementation, replace with your criteria)
def evaluate_mesh_quality(verts, faces):
    return len(verts) * len(faces)  # Placeholder: use a real quality metric here


# Function to generate the best 3D mesh
def generate_best_3d_obj(model, embed_fn, output_path):
    # Parameters
    resolution = 128
    bounding_box = [[-1, 1], [-1, 1], [-1, 1]]

    # Sample 3D points
    points = sample_3d_points(resolution, bounding_box)

    # Query the NeRF model
    raw = query_nerf(model, points, embed_fn)

    # Extract density
    density = tf.nn.relu(raw[..., 3]).numpy()

    # Check the range of density values
    density_min, density_max = density.min(), density.max()

    # Grid search for the best density threshold
    best_threshold = None
    best_quality = float('-inf')  # Initialize with negative infinity
    best_verts = None
    best_faces = None

    thresholds = np.linspace(density_min, density_max, num=10)

    for threshold in thresholds:
        try:
            verts, faces = density_to_mesh(density, threshold)
            quality = evaluate_mesh_quality(verts, faces)  # Implement this function based on your criteria
            if quality > best_quality:
                best_quality = quality
                best_threshold = threshold
                best_verts = verts
                best_faces = faces
        except Exception as e:
            print(f"Failed to generate mesh at threshold {threshold}: {e}")

    if best_verts is not None and best_faces is not None:
        save_mesh_to_obj(best_verts, best_faces, output_path)
        print(f"Best threshold: {best_threshold}, Best quality: {best_quality}")
    else:
        print("No valid mesh found.")


def get_poses_images(reconstruction):
    poses = []
    images = []
    new_shape = (100, 100)

    # Iterate through images and access their poses and filenames
    for _, image in reconstruction.images.items():
        cam_from_world = image.cam_from_world
        qvec = cam_from_world.rotation.quat
        rotation_matrix = quaternion_to_rotation_matrix(qvec)
        
        # Store the pose and the corresponding image name
        poses.append(rotation_matrix)

        image_path = image_dir / image.name
        img = PILImage.open(image_path)
        image = np.array(img, dtype=np.float32) / 255.0
        image = resize(image, new_shape, order=1, mode='reflect')
        images.append(image)

    return images ,poses


def get_focal():
    total_focal = 0
    count = 0
    for camera_id ,camera in reconstruction.cameras.items():
        total_focal += camera.params[0]
        count += 1

    focal = total_focal / count

    return focal


def quaternion_to_rotation_matrix(qvec):
    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2), 0],
        [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1), 0],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2), 0],
        [0, 0, 0, 1]
    ])


def save_frame(frame, frame_count, output_folder):
    frame_image = PILImage.fromarray(frame)
    frame_filename = output_folder / f"frame_{frame_count}.jpg"
    frame_image.save(frame_filename, format='JPEG')


def extract_and_save_frames(video_path, output_folder, start_time, end_time, frame_interval):
    # Load the video
    clip = VideoFileClip(video_path)
    
    # Calculate the duration and start frame extraction at the specified start time
    fps = clip.fps
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Create the output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Extract frames and save every nth frame
    frame_count = 0
    saved_frame_count = 0

    for frame in clip.iter_frames(fps=fps):
        current_time = frame_count / fps

        if current_time < start_time:
            frame_count += 1
            continue
        if current_time > end_time:
            break
        
        if frame_interval == 0 or frame_count % frame_interval == 0:
            save_frame(frame, frame_count, output_folder)
            saved_frame_count += 1

        frame_count += 1
    
    print(f"Total frames saved: {saved_frame_count}")


