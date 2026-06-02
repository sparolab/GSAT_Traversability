import os
import sys
import random
import shutil
import argparse
import yaml

def split_dataset(points_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.3, eval_ratio=0.0, seed=42):
    file_names = sorted([f for f in os.listdir(points_dir) if f.endswith('.bin')])

    random.seed(seed)
    random.shuffle(file_names)
    
    total = len(file_names)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    eval_count = total - train_count - val_count

    splits = {
        'train': file_names[:train_count],
        'val': file_names[train_count:train_count+val_count],
        'test': file_names[train_count+val_count:]
    }

    for split_name, files in splits.items():
        points_out_dir = os.path.join(output_dir, split_name, 'point')
        labels_out_dir = os.path.join(output_dir, split_name, 'label')
        os.makedirs(points_out_dir, exist_ok=True)
        os.makedirs(labels_out_dir, exist_ok=True)
        
        for idx, file_name in enumerate(sorted(files)):
            src_points = os.path.join(points_dir, file_name)
            src_labels = os.path.join(labels_dir, file_name)
            
            if not os.path.exists(src_labels):
                print(f"[{split_name}] Skipping {file_name}: matching label file not found.")
                continue

            new_name = f"{idx:06d}.bin"
            dst_points = os.path.join(points_out_dir, new_name)
            dst_labels = os.path.join(labels_out_dir, new_name)
            
            shutil.copy(src_points, dst_points)
            shutil.copy(src_labels, dst_labels)
            
            print(f"[{split_name}] Copied {src_points} -> {dst_points}")
            print(f"[{split_name}] Copied {src_labels} -> {dst_labels}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./../data_tools/config/data_split.yaml')
    parser.add_argument('--key', type=str, default='hill_example',
                        help='top-level config key to use (e.g. hill_example)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.key not in config:
        print(f"[Error] config key '{args.key}' not found in {args.config}.")
        sys.exit(1)
    cfg = config[args.key]

    root_dir = cfg.get("root_dir", "/data/root_dir")
    preprocess_dir = cfg.get("preprocess_dir", "/data/output_root")
    save_dir = cfg.get("save_dir", "/data/output_root")

    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.25)
    test_ratio = cfg.get("test_ratio", 0.05)

    points_dir = os.path.join(root_dir, preprocess_dir,"point")
    labels_dir = os.path.join(root_dir, preprocess_dir, "label")
    output_dir = os.path.join(root_dir, save_dir)

    split_dataset(points_dir, labels_dir, output_dir, train_ratio, val_ratio, test_ratio)
