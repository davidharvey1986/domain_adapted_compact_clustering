import glob
import os
import pickle
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from collections import defaultdict

def get_cross_section_from_filename(filename):
    base_name = os.path.basename(filename)
    name = os.path.splitext(base_name)[0].lower()
    if "cdm" in name or "flamingo" in name:
        return 0.0
    matches = re.findall(r"_(\d+\.\d+|\d+)", name)
    if matches:
        return float(matches[0])
    return 0.0

def cross_section_to_binary_label(cross_sections):
    return (cross_sections > 0).long()

class BalancedClusterDataset(Dataset):
    def __init__(self, data_dir, file_pattern, transform=None, normalization_stats=None, args=None):
        self.data = []
        self.cross_sections = []
        self.transform = transform
        self.normalization_stats = normalization_stats
        self.args = args
        self.file_indices = []
        self.files_data = {}
        
        self.by_img_idx = defaultdict(list)
        
        files = glob.glob(os.path.join(data_dir, file_pattern))
        if not files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in directory '{data_dir}'")

        print(f"Loading {len(files)} files matching '{file_pattern}'...")

        for file_idx, file_path in enumerate(files):
            with open(file_path, "rb") as f:
                meta, images = pickle.load(f)
                cross_section = get_cross_section_from_filename(file_path)
                images = images[:, 0:1]
                
                if args and args.use_log_transform:
                    images = np.log10(images * meta['norms'][:, 0:1, None, None])
                
                self.files_data[file_idx] = {
                    'images': images,
                    'cross_section': cross_section,
                    'filename': os.path.basename(file_path)
                }
                
                for img_idx in range(len(images)):
                    global_idx = len(self.data)
                    self.data.append(images[img_idx])
                    self.cross_sections.append(cross_section)
                    self.file_indices.append((file_idx, img_idx))
                    self.by_img_idx[img_idx].append(global_idx)

        self.data = np.array(self.data)
        self.cross_sections = np.array(self.cross_sections)

        print(f"Data shape: {self.data.shape}")
        print(f"Unique cross-sections: {np.unique(self.cross_sections)}")

        binary_labels = cross_section_to_binary_label(torch.tensor(self.cross_sections))
        unique, counts = torch.unique(binary_labels, return_counts=True)
        print(f"Classification class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx.item()}: {count.item()} samples ({count.item()/len(binary_labels)*100:.1f}%)")

    def __len__(self):
        return len(self.cross_sections)

    def __getitem__(self, idx):
        image = self.data[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        if self.normalization_stats is not None and self.args and self.args.use_normalization:
            mean, std = self.normalization_stats
            image_tensor = (image_tensor - mean) / std

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
            
        cross_section = torch.tensor(self.cross_sections[idx], dtype=torch.float32)
        file_idx, img_idx = self.file_indices[idx]

        binary_label = cross_section_to_binary_label(cross_section.unsqueeze(0)).squeeze(0)
        return image_tensor, cross_section, binary_label, file_idx, img_idx

    
    def get_same_position_candidates(self, file_idx, img_idx):
        candidates = []
        for global_idx in self.by_img_idx[img_idx]:
            cand_file_idx, _ = self.file_indices[global_idx]
            if cand_file_idx != file_idx:
                candidates.append(global_idx)
        return candidates
        
    @staticmethod
    def compute_normalization_stats(dataset):
        all_data = []
        if isinstance(dataset, Subset):
            original_dataset = dataset.dataset
            indices = dataset.indices
            for idx in indices:
                all_data.append(original_dataset.data[idx])
        else:
            for idx in range(len(dataset)):
                all_data.append(dataset.data[idx])
        
        all_data = np.stack(all_data)
        mean = np.mean(all_data)
        std = np.std(all_data)
        
        if std == 0:  
            std = 1.0
                
        return mean, std

def create_balanced_test_set(dataset, test_size=0.2, random_state=42):
    binary_labels = cross_section_to_binary_label(torch.tensor(dataset.cross_sections))
    
    class_0_indices = torch.where(binary_labels == 0)[0].numpy()
    class_1_indices = torch.where(binary_labels == 1)[0].numpy()
    
    print(f"Class distribution:")
    print(f"  Class 0: {len(class_0_indices)} samples")
    print(f"  Class 1: {len(class_1_indices)} samples")
    
    total_test_samples = int(test_size * len(dataset))
    samples_per_class = total_test_samples // 2
    samples_per_class = min(samples_per_class, len(class_0_indices), len(class_1_indices))
    
    print(f"Creating balanced test set with {samples_per_class} samples per class ({samples_per_class * 2} total)")
    
    np.random.seed(random_state)
    test_0_indices = np.random.choice(class_0_indices, size=samples_per_class, replace=False)
    test_1_indices = np.random.choice(class_1_indices, size=samples_per_class, replace=False)
    
    test_indices = np.concatenate([test_0_indices, test_1_indices])
    np.random.shuffle(test_indices)
    
    test_indices = test_indices.tolist()
    
    test_labels = binary_labels[test_indices]
    test_unique, test_counts = torch.unique(test_labels, return_counts=True)
    print(f"Balanced test set distribution:")
    for class_idx, count in zip(test_unique, test_counts):
        print(f"  Class {class_idx.item()}: {count.item()} samples ({count.item()/len(test_labels)*100:.1f}%)")
    
    return test_indices

def prepare_dataloaders(args):
    data_loaders = {}

    domain_patterns = {
        "bahamas": "bahamas*.pkl",
        "darkskies": "darkskies*.pkl",
    }

    if args.source_domain not in domain_patterns or args.target_domain not in domain_patterns:
        raise ValueError(f"Source and target domains must be one of: {list(domain_patterns.keys())}")

    source_pattern = domain_patterns[args.source_domain]
    target_pattern = domain_patterns[args.target_domain]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(args.aug_h_flip_prob),
        transforms.RandomVerticalFlip(args.aug_v_flip_prob),
        transforms.RandomApply([transforms.RandomRotation(degrees=args.aug_rotation_degrees, fill=0)], p=args.aug_rotation_prob),
        transforms.RandomApply([
            transforms.RandomResizedCrop(size=(args.image_size, args.image_size), 
                                       scale=(args.aug_crop_scale_min, args.aug_crop_scale_max), 
                                       ratio=(1.0, 1.0))
        ], p=args.aug_crop_prob),
    ])
    
    temp_source_dataset = BalancedClusterDataset(args.data_dir, source_pattern, args=args)

    print(f"Source domain setup:")
    source_test_indices = create_balanced_test_set(
        temp_source_dataset, 
        test_size=(1 - args.train_split),
        random_state=args.seed
    )
    
    all_source_indices = set(range(len(temp_source_dataset)))
    source_test_indices_set = set(source_test_indices)
    source_train_indices = list(all_source_indices - source_test_indices_set)
    
    print(f"  Training samples: {len(source_train_indices)}")
    print(f"  Test samples: {len(source_test_indices)}")

    temp_source_train_dataset = Subset(temp_source_dataset, source_train_indices)
    source_stats = BalancedClusterDataset.compute_normalization_stats(temp_source_train_dataset)
    print(f"Source dataset normalization - Mean: {source_stats[0]:.4f}, Std: {source_stats[1]:.4f}")
    
    source_dataset_train = BalancedClusterDataset(args.data_dir, source_pattern, transform=train_transform, 
                                                 normalization_stats=source_stats, args=args)
    source_dataset = BalancedClusterDataset(args.data_dir, source_pattern, normalization_stats=source_stats, 
                                           args=args)
    
    target_dataset_train = BalancedClusterDataset(args.data_dir, target_pattern, transform=train_transform, 
                                                 normalization_stats=source_stats, args=args)
    target_dataset = BalancedClusterDataset(args.data_dir, target_pattern, normalization_stats=source_stats, 
                                           args=args)

    source_train_dataset = Subset(source_dataset_train, source_train_indices)
    source_test_dataset = Subset(source_dataset, source_test_indices)

    print(f"Target domain setup:")
    target_test_indices = create_balanced_test_set(
        target_dataset, 
        test_size=(1 - args.train_split),
        random_state=args.seed
    )
    
    all_target_indices = set(range(len(target_dataset)))
    target_test_indices_set = set(target_test_indices)
    target_train_indices = list(all_target_indices - target_test_indices_set)
    
    print(f"  Training samples: {len(target_train_indices)}")
    print(f"  Test samples: {len(target_test_indices)}")
    
    target_train_dataset = Subset(target_dataset_train, target_train_indices)
    target_test_dataset = Subset(target_dataset, target_test_indices)

    data_loaders["source_train"] = DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=True
    )

    data_loaders["source_val"] = DataLoader( 
        source_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    data_loaders["target_train"] = DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=True
    )

    data_loaders["target_test"] = DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"DataLoader summary:")
    print(f"  Source train: {len(source_train_dataset)} samples")
    print(f"  Source test: {len(source_test_dataset)} samples") 
    print(f"  Target train: {len(target_train_dataset)} samples")
    print(f"  Target test: {len(target_test_dataset)} samples")

    return data_loaders