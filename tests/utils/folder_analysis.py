import os
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

def analyze_sunrgbd_folder(root_dir, sample_limit=3):
    """
    Analyze SUN RGB-D folder structure and data types.
    
    Args:
        root_dir (str): path like data/sunrgbd/kv1/b3dodata
        sample_limit (int): how many img_xxxx folders to sample deeply
    """
    print(f"\n📁 Analyzing dataset at: {root_dir}\n")
    
    img_dirs = sorted([
        d for d in os.listdir(root_dir)
        if d.startswith("img_") and os.path.isdir(os.path.join(root_dir, d))
    ])
    
    print(f"Total image folders found: {len(img_dirs)}")
    print("Sample folders:", img_dirs[:5], "\n")

    modality_summary = {}

    for img_dir in img_dirs[:sample_limit]:
        print(f"\n{'='*60}")
        print(f"🔍 Inspecting {img_dir}")
        print(f"{'='*60}")

        full_path = os.path.join(root_dir, img_dir)
        contents = os.listdir(full_path)

        for item in contents:
            item_path = os.path.join(full_path, item)

            modality_summary.setdefault(item, 0)
            modality_summary[item] += 1

            if os.path.isdir(item_path):
                print(f"\n📂 Folder: {item}")
                files = os.listdir(item_path)
                print(f"  Files ({len(files)}): {files[:5]}{'...' if len(files) > 5 else ''}")

                # --- Inspect common known folders ---
                if item == "image":
                    inspect_rgb_image(item_path)
                elif "depth" in item:
                    inspect_depth(item_path)
                elif "annotation" in item:
                    inspect_annotation(item_path)

            else:
                print(f"\n📄 File: {item}")
                inspect_file(item_path)

    print(f"\n\n📊 GLOBAL MODALITY SUMMARY")
    print("="*60)
    for k, v in modality_summary.items():
        print(f"{k:<25} appears in {v} folders")


def inspect_file(path):
    if path.endswith('.txt'):
        with open(path, 'r', errors='ignore') as f:
            lines = f.readlines()
        print(f"  📄 Text file | Lines: {len(lines)}")
        print("  First 2 lines:", lines[:2])

    elif path.endswith('.mat'):
        mat = loadmat(path)
        print(f"  📦 MAT file | Keys: {list(mat.keys())}")

    else:
        print("  📄 Unknown file type")

def inspect_annotation(path):
    files = os.listdir(path)
    print(f"  🧾 Annotation files: {files[:5]}{'...' if len(files) > 5 else ''}")


def inspect_depth(path):
    files = os.listdir(path)
    depth_files = [f for f in files if f.endswith(('.png', '.mat'))]
    
    if not depth_files:
        return

    f = depth_files[0]
    p = os.path.join(path, f)

    if f.endswith('.png'):
        depth = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        print(f"  🌊 Depth PNG: {f} | Shape: {depth.shape} | dtype: {depth.dtype}")
        print(f"     min={depth.min()}, max={depth.max()}")
    elif f.endswith('.mat'):
        mat = loadmat(p)
        print(f"  🌊 Depth MAT: {f} | Keys: {list(mat.keys())}")


def inspect_rgb_image(path):
    imgs = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
    if not imgs:
        return
    img_path = os.path.join(path, imgs[0])
    img = cv2.imread(img_path)
    print(f"  🖼 RGB Image: {imgs[0]} | Shape: {img.shape} | dtype: {img.dtype}")
