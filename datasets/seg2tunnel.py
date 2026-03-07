import os
import numpy as np
import torch
from .defaults import DefaultDataset_new

class Seg2TunnelDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/Normalised",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        skip=1,
        use_random_clicks=False,
        num_prompt_points=8,
        num_object_points=5,
        overfit=False,
        use_centroid=False,
        ignore_index=0,
    ):
        self.split = split
        self.data_root = data_root
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.overfit = overfit
        self.use_centroid = use_centroid
        self.ignore_index = ignore_index
        self.skip = skip
        self.use_random_clicks = use_random_clicks

        # Fetch data list before super() initialization, mirroring KITTI360
        self.data_list = self.get_data_list()

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        if self.overfit:
            self.data_list = self.data_list[:10]

    def get_data_list(self):
        """Get list of txt files for the dataset"""
        split_path = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_path):
            # Fallback in case train/val/test splits are strictly in the root directory
            split_path = self.data_root
            
        data_list = sorted([f for f in os.listdir(split_path) if f.endswith('.txt')])
        return data_list

    def get_data(self, idx):
        """Load point cloud from txt file"""
        split_path = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_path):
            split_path = self.data_root
            
        txt_file = os.path.join(split_path, self.data_list[idx])
        data = np.loadtxt(txt_file)
        
        # ==========================================
        # EXTREME VRAM SAVER: Random Downsampling
        # ==========================================
        # Applied ONLY during training to preserve validation accuracy
        if self.split == "train":
            max_points = 16384  
            if len(data) > max_points:
                indices = np.random.choice(len(data), max_points, replace=False)
                data = data[indices]
        # ==========================================

        coord = data[:, :3].astype(np.float32)
        strength = data[:, 3:4].astype(np.float32)  # Extracted as (N, 1) for 1-channel feature backbone
        segment = data[:, 4].astype(np.int32)       
        instance = segment.copy() # Treating semantic class essentially as instance for masking logic
        
        # Aligned dict format: explicitly locking condition and domain
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            instance=instance,
            condition="Seg2Tunnel",  # Ensures correct task mapping in SNAP
            domain="Tunnel"          # Natively bypasses the PTV3 AssertionError
        )
        return data_dict

    def process_data(self, data_dict):
        """
        Extract variables mimicking KITTI360 process_data flow.
        Responsible for generating object masks and prompt points.
        """
        labels = data_dict['instance']
        coord = data_dict['coord']
        grid_coord = data_dict.get('grid_coord', coord) 
        strength = data_dict['strength']
        condition = data_dict['condition']
        domain = data_dict['domain']
        segment = data_dict['segment']

        instance = data_dict.get('instance', segment)

        masks = []
        mask_labels = []
        prompt_points = []

        unique_labels = np.unique(labels)
        
        # Filter out the ignore index (e.g., 0 for background/noise)
        unique_labels = unique_labels[unique_labels != self.ignore_index]
        
        # Safe return if empty (prevents crashing on rings that only contain background noise)
        if len(unique_labels) == 0:
            data_dict['point'] = torch.empty((0, self.num_object_points, 3), dtype=torch.float32)
            data_dict['masks'] = torch.empty((0, len(labels)), dtype=torch.long)
            data_dict['mask_labels'] = np.array([])
            return data_dict

        np.random.shuffle(unique_labels)
        
        if self.split == "train":
            num_obj = min(self.random_points, len(unique_labels))
        else:
            num_obj = len(unique_labels)

        for i in range(num_obj):
            label = unique_labels[i]
            point_idxs = np.where(labels == label)[0]
            
            # Skip statistically insignificant clusters during validation
            if self.split == "val" and len(point_idxs) < 10:
                continue
                
            seg_label = segment[point_idxs[0]] # Retrieve semantic class
            obj_coord = coord[point_idxs]

            # Centralized point sampling logic mimicking KITTI360
            if self.use_centroid:
                centroid = np.mean(obj_coord, axis=0)
                distances = np.linalg.norm(obj_coord - centroid, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = obj_coord[closest_idx]

                remaining_idxs = np.delete(np.arange(len(obj_coord)), closest_idx)
                if len(remaining_idxs) >= self.num_object_points - 1:
                    sampled_idxs = np.random.choice(remaining_idxs, self.num_object_points - 1, replace=True)
                    sampled_points = np.vstack([closest_point, obj_coord[sampled_idxs]])
                else:
                    sampled_points = obj_coord[np.random.choice(len(obj_coord), self.num_object_points, replace=True)]
            else:
                sampled_points = obj_coord[np.random.choice(len(obj_coord), self.num_object_points, replace=True)]

            prompt_points.append(sampled_points)

            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            
            # Important: Cross-Entropy loss expects labels starting from 0. 
            # If your dataset's labels in the text file are 1-6, subtracting 1 correctly maps them to 0-5.
            mask_labels.append(seg_label - 1) 

        # Final sanity check before array conversion in case all valid labels were skipped
        if len(masks) == 0:
            data_dict['point'] = torch.empty((0, self.num_object_points, 3), dtype=torch.float32)
            data_dict['masks'] = torch.empty((0, len(labels)), dtype=torch.long)
            data_dict['mask_labels'] = np.array([])
            return data_dict

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)

        # Recreate dictionary strictly enforcing PyTorch Tensor constraints
        data_dict = dict(
            coord=coord, 
            grid_coord=grid_coord, 
            strength=strength, 
            point=torch.from_numpy(prompt_points).float(), 
            condition=condition, 
            domain=domain,
            masks=torch.from_numpy(masks).long(), 
            mask_labels=mask_labels, 
            segment=segment,      
            instance=instance
        )

        return data_dict

    def class_labels(self):
        """
        Maps the network's internal 0-5 indices back to your structural 
        S1-S6 definitions for evaluation and visualization.
        """
        class_labels = {
            0: "Key segment ",
            1: "Adjacent segment left ",
            2: "Standard segment left ",
            3: "Standard segment middle ",
            4: "Standard segment right ",
            5: "Adjacent segment right ",
        }
        return class_labels