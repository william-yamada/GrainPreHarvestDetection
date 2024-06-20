import argparse
import json
import yaml
from glob import glob
import os
import cv2
import tqdm
import numpy as np

# global variables
def image_list(image_dir):
    l = glob(os.path.join(image_dir, '*.jpg'))
    print("Loading images from:", image_dir)
    print("Number of images:", len(l))
    return l

def get_images_annotations(image_file, images, annotations, is_docker):
    annotation_file = image_file.replace('images', 'labels').replace('.jpg', '.txt')
    try:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Annotation file {annotation_file} not found.")
        return

    # get width and height of the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image {image_file}.")
        return

    height, width, _ = image.shape

    image_entry = {
        "file_name": image_file.replace("/home/william/Projects/GrainPreHarvestDetection/", "/usr/src/ultralytics/") if is_docker else image_file,
        "height": height,
        "width": width,
        "id": len(images) + 1,
    }
    images.append(image_entry)

    for line in lines:
        line = line.strip().split()
        class_index = int(line[0])
        segmentations = []
        min_x, min_y, max_x, max_y = width, height, 0, 0
        for i in range(1, len(line), 2):
            x = float(line[i]) * width
            y = float(line[i + 1]) * height
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            segmentations.append([x, y])

        if len(segmentations) < 3:
            continue

        segmentations_np = np.array(segmentations, dtype=np.float32)
        area = cv2.contourArea(segmentations_np)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        annotation_entry = {
            "id": len(annotations) + 1,
            "bbox": bbox,
            "bbox_mode": 0,
            "category_id": class_index,
            "iscrowd": 0,
            "segmentation": [segmentations],
            "image_id": image_entry["id"],
            "area": area,
        }
        annotations.append(annotation_entry)

def save_coco_format(images, annotations, categories, output_file):
    coco_data = {
        "info": {
            "description": "Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-01-01 00:00:00.000000",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to the dataset configuration file')
    parser.add_argument('--docker', action='store_true', help='Flag to indicate if the script is running inside a docker container')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)

    images = []
    annotations = []
    categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(data['names'])]

    for split in ['train', 'val', 'test']:
        print(f"Converting {split} data...")
        image_dir = data[split]
        image_files = image_list(image_dir)
        for image_file in tqdm.tqdm(image_files):
            get_images_annotations(image_file, images, annotations, args.docker)
        print(f"Saving {split} data...")
        output_file = image_dir.replace('images', 'coco_docker.json') if args.docker else image_dir.replace('images', 'coco.json')
        save_coco_format(images, annotations, categories, output_file)
        images = []
        annotations = []

    print("Conversion completed!")
