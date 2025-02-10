"""
YOLO Training Script for Image Segmentation

This script provides a command-line interface for training YOLO models for image segmentation tasks.
It handles dataset splitting, YAML configuration generation, model training, and evaluation.

Usage:
python standalone_yolo.py \
    --real-dir /path/to/customer_data \
    --syn-dir /path/to/advex_data \
    --epochs 200 \
    --augment True \ # remove this to disable augmentations
    --seed 1

The real and synthetic data directories should contain:
- Images (.jpg, .jpeg, or .png)
- Corresponding masks (.jpg, .jpeg, or .png)

"""

import os
import shutil
from pathlib import Path
import random
from typing import Tuple
import yaml
from ultralytics import YOLO
import click
import tempfile
from contextlib import contextmanager
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import settings as ultralytics_settings

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]


def apply_augmentations(image, mask):
    """Apply augmentations such as rotation, crop, and jitter to both image and mask."""
    angle = random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_mask = cv2.warpAffine(mask, M, (w, h))
    return rotated_image, rotated_mask

def augment_dataset(train_dir, target_count) -> str:
    """Apply augmentations to real dataset to create an augmented dataset with target_count images."""
    img_train_dir = os.path.join(train_dir, "images")
    mask_train_dir = os.path.join(train_dir, "masks")
    augmented_img_dir = os.path.join(train_dir, "augmented", "images")
    augmented_mask_dir = os.path.join(train_dir, "augmented", "masks")
    augmented_dir = os.path.join(train_dir, "augmented") # path to augmented dataset

    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
        
    os.makedirs(augmented_img_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)
    
    real_images = [img for img in os.listdir(img_train_dir) if img.endswith(tuple(SUPPORTED_IMAGE_FORMATS))]
    num_real = len(real_images)
    num_augmentations_needed = max(0, target_count - num_real)
    
    augmented_images = 0

    with tqdm(total=num_augmentations_needed, desc="Augmenting dataset") as pbar:
        while augmented_images < num_augmentations_needed:
            img_name = random.choice(real_images)
            img_path = os.path.join(img_train_dir, img_name)
            mask_path = os.path.join(mask_train_dir, img_name)
            
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                click.echo(f"Warning: Failed to load image or mask for {img_name}, skipping.")
                continue
            augmented_image, augmented_mask = apply_augmentations(image, mask)
            
            aug_img_name = f"aug_{augmented_images}_{img_name}"
            cv2.imwrite(os.path.join(augmented_img_dir, aug_img_name), augmented_image)
            cv2.imwrite(os.path.join(augmented_mask_dir, aug_img_name), augmented_mask)
            augmented_images += 1
            pbar.update(1)
    return augmented_dir


def mask_to_polygons(mask):
    """Convert a mask ndarray to polygon vertices."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:  # Need at least 3 points for a valid polygon
            continue
        # Extract x, y coordinates from the contour
        poly = approx.squeeze().reshape(-1).tolist()
        polygons.append(poly)
    return polygons


def unique_objects(mask, kernel_size: int = 3):
    """Use connected components to find unique objects in the mask"""
    # convert to binary
    mask = (mask > 0).astype(np.uint8)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Apply erosion to remove small artifacts
        mask = cv2.erode(mask, kernel, iterations=1)
        # Apply dilation to restore object shapes
        mask = cv2.dilate(mask, kernel, iterations=1)
    # use when mask is binary
    num_objects, labels = cv2.connectedComponents(mask)

    return num_objects, labels


def convert_real_to_yolo(folder_path, output_folder_path: str, leave_images_untouched: bool) -> dict:
    images_dir = os.path.join(folder_path, "images")
    masks_dir = os.path.join(folder_path, "masks")
    assert os.path.exists(images_dir)
    assert os.path.exists(masks_dir)

    # Copy image directory
    if not leave_images_untouched:
        shutil.copytree(images_dir, os.path.join(output_folder_path, "images"))

    output_folder = os.path.join(output_folder_path, "labels")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_id = 1

    for filename in os.listdir(images_dir):
        if not any(filename.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
            continue

        base_filename = os.path.splitext(filename)[0]

        image_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        image = Image.open(image_path)
        image_width, image_height = image.size
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # mask = (mask > 0).astype(np.uint8) * 255
        unique_objects_count, labels = unique_objects(mask)
        # if there are no unique objects, create an empty file
        if unique_objects_count == 1:
            print(f"Warning: Filename {base_filename} contains no objects. Creating empty file.")
            with open(os.path.join(output_folder, f"{base_filename}.txt"), "w") as f:
                f.write("")
            continue

        for obj_id in range(1, unique_objects_count):  # 0 = background.
            obj_mask = (labels == obj_id).astype(np.uint8) * 255

            # Convert mask to polygons
            polygons = mask_to_polygons(obj_mask)
            for poly in polygons:

                if (
                    len(poly) < 6
                ):  # Polygons have at least 3 points (and hence 6 coordinates x,y)
                    continue

                yolo_segmentation = [
                    f"{(x) / image_width:.5f} {(y) / image_height:.5f}"
                    for x, y in zip(poly[::2], poly[1::2])
                ]
                yolo_segmentation = " ".join(yolo_segmentation)

                # Generate the YOLO segmentation annotation line
                yolo_annotation = f"0 {yolo_segmentation}"

                # Save the YOLO segmentation annotation in a file
                output_filename = os.path.join(output_folder, f"{base_filename}.txt")
                with open(output_filename, "a+") as file:
                    file.write(yolo_annotation + "\n")

        image_id += 1


def create_dataset_yaml(
    train_dir: str, val_dir: str, test_dir: str, save_path: str, num_classes: int = 1
) -> None:
    """
    Create YAML configuration file for YOLO training.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        num_classes: Number of classes in the dataset
        save_path: Path where to save the YAML file
    """
    data = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
        "nc": num_classes,
    }

    with open(save_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


@contextmanager
def temporary_split_directory():
    """
    Context manager that creates a temporary directory for dataset splits
    and cleans it up afterwards.

    Yields:
        str: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def split_data(
    source_dir: str,
    temp_dir: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> Tuple[str, str, str]:
    """
    Split dataset into train, validation and test sets.

    Args:
        source_dir: Directory containing images and labels
        temp_dir: Temporary directory to store split datasets
        split_ratios: Tuple of (train, val, test) ratios that sum to 1.0

    Returns:
        Tuple[str, str, str]: Paths to train, validation, and test directories

    The function expects:
    1. Images in supported formats (.jpg, .jpeg, .png)
    2. Labels in YOLO format (.txt) with matching filenames
    3. Both images and labels in the same source directory
    """
    # Create output directories
    train_dir = os.path.join(temp_dir, "train")
    val_dir = os.path.join(temp_dir, "val")
    test_dir = os.path.join(temp_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        images_dir = os.path.join(dir_path, "images")
        labels_dir = os.path.join(dir_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    image_src_dir = os.path.join(source_dir, "images")
    label_src_dir = os.path.join(source_dir, "labels")
    # Get all image files
    image_files = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_files.extend(list(Path(image_src_dir).glob(f"*{ext}")))

    # Randomly shuffle files
    random.shuffle(image_files)

    # Calculate split sizes
    total = len(image_files)
    train_size = int(total * split_ratios[0])
    val_size = int(total * split_ratios[1])

    # Split into train, val, test
    train_files = image_files[:train_size]
    val_files = image_files[train_size : train_size + val_size]
    test_files = image_files[train_size + val_size :]

    # Copy files to respective directories
    for files, dir_path in [
        (train_files, train_dir),
        (val_files, val_dir),
        (test_files, test_dir),
    ]:
        for img_path in files:
            # Copy image
            shutil.copy2(img_path, os.path.join(dir_path, "images", img_path.name))

            # Copy corresponding label file
            label_path_name = os.path.basename(img_path.with_suffix(".txt"))
            label_path = os.path.join(label_src_dir, label_path_name)
            shutil.copy2(label_path, os.path.join(dir_path, "labels", label_path_name))

    return train_dir, val_dir, test_dir


def train_yolo(data_yaml_path: str, epochs: int = 200) -> YOLO:
    """
    Train YOLO model for segmentation.

    Args:
        data_yaml_path: Path to YAML file containing dataset configuration
        epochs: Number of training epochs

    Returns:
        YOLO: Trained YOLO model instance

    The function uses YOLOv8n-seg as the base model and trains it with:
    - Image size: 1024x1024
    - Checkpoints saved in 'yolo_training/segmentation_model'
    """
    # Load a model
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=1024,
        save=True,  # Save checkpoints
        project="yolo_training",
        name="segmentation_model",
    )

    return model


def evaluate_yolo(model: YOLO):
    """
    Evaluate YOLO model on test set.

    Args:
        model: Trained YOLO model instance

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Run evaluation
    metrics_ultralytics = model.val(split="test", verbose=False)

    metrics = {
        "Average Precision": metrics_ultralytics.seg.map50,
        "Average Recall": metrics_ultralytics.seg.mr,
        "F1" : metrics_ultralytics.seg.f1.mean(),
    }
    return metrics


def add_synthetic_data(train_dir: str, syn_dir: str):
    """
    Add synthetic data to train set
    """
    
    img_train_dir = os.path.join(train_dir, "images")
    label_train_dir = os.path.join(train_dir, "labels")
    img_syn_dir = os.path.join(syn_dir, "images")
    label_syn_dir = os.path.join(syn_dir, "labels")
    for syn_img_path in os.listdir(img_syn_dir):
        syn_label_path = f"{os.path.splitext(syn_img_path)[0]}.txt"
        shutil.copy2(
            os.path.join(img_syn_dir, syn_img_path),
            os.path.join(img_train_dir, syn_img_path),
        )
        shutil.copy2(
            os.path.join(label_syn_dir, syn_label_path),
            os.path.join(label_train_dir, syn_label_path),
        )


@click.command()
@click.option(
    "--real-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing real images and labels",
)
@click.option(
    "--syn-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing synthetic images and labels",
)
@click.option(
    "--augment",
    is_flag=False,
    help="Apply augmentations to real data",
)
@click.option(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs for training",
    show_default=True,
)
@click.option(
    "--seed",
    default=1,
    type=int,
    help="Random seed for reproducibility",
    show_default=True,
)
def main(real_dir: str, syn_dir: str, augment: bool, epochs: int, seed: int):
    """
    Train YOLO model for segmentation.

    This command-line interface handles the complete training pipeline:
    1. Splits the dataset into train/val/test sets in a temporary directory
    2. Creates YAML configuration
    3. Trains the YOLO model
    4. Evaluates the model on test set
    5. Cleans up temporary files
    """
    # Disable W&B logging
    ultralytics_settings.update({"wandb": False})
    # Set random seed for reproducibility
    random.seed(seed)

    # Use temporary directory for dataset splits
    with temporary_split_directory() as temp_dir:
        click.echo("Converting real data to YOLO format...")
        convert_real_to_yolo(real_dir, temp_dir, leave_images_untouched=False)
        click.echo("Splitting dataset...")
        train_dir, val_dir, test_dir = split_data(temp_dir, temp_dir)

        # Create dataset YAML in the output directory
        yaml_path = os.path.join(temp_dir, "dataset.yaml")
        create_dataset_yaml(
            train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, save_path=yaml_path
        )

        click.echo("Starting training real model...")
        real_model = train_yolo(yaml_path, epochs=epochs)

        click.echo("Evaluating real model...")
        real_results = evaluate_yolo(real_model)
        
        augmented_results = None
        if augment:
            click.echo("Applying augmentations and training on real + augmented data...")
            total_target_count = len(os.listdir(os.path.join(real_dir, "images"))) + len(os.listdir(os.path.join(syn_dir, "images")))
            augmented_dir = augment_dataset(real_dir, total_target_count)
            # convert augmented data to YOLO format
            convert_real_to_yolo(augmented_dir, augmented_dir, leave_images_untouched=True)
            augmented_model = train_yolo(yaml_path, epochs)
            augmented_results = evaluate_yolo(augmented_model)

        click.echo("Adding synthetic data to train set")
        if not os.path.exists(os.path.join(syn_dir, "labels")):
            click.echo(f"Labels directory does not exist in {syn_dir}. Creating...")
            convert_real_to_yolo(syn_dir, syn_dir, leave_images_untouched=True)
            
        add_synthetic_data(train_dir, syn_dir)

        click.echo("Starting training synthetic model...")
        syn_model = train_yolo(yaml_path, epochs=epochs)

        click.echo("Evaluating synthetic model...")
        syn_results = evaluate_yolo(syn_model)

        combined_results = {
            "Customer Data": real_results,
            "Customer Data + Advex": syn_results,
        }
        
        if augment:
            combined_results["Customer Data + Augmentations"] = augmented_results
            
        # Print results
        click.echo("--------------------------------")
        click.echo("Evaluation results:")
        for technique, results in combined_results.items():
            click.echo(f"{technique}:")
            for metric, value in results.items():
                click.echo(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    
    main()
