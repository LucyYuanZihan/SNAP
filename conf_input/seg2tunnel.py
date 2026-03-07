"""
Configuration for Seg2Tunnel Level II dataset input pipeline
"""

# Dataset settings
dataset_type = "Seg2TunnelDataset"
data_root = "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/Normalised"

# These represent the 0-indexed structural outputs of the network
class_labels = {
    0: "Key segment",
    1: "Adjacent segment left",
    2: "Standard segment left",
    3: "Standard segment middle",
    4: "Standard segment right",
    5: "Adjacent segment right",
}

# Maps Raw .txt labels -> Network labels (0-indexed)
# 0 is background noise, mapped to -1 to be ignored by loss functions
learning_map = {
    0: -1,  # ignore_index for noise/non-structural elements
    1: 0,   # key segment
    2: 1,   # adjacent segment left
    3: 2,   # standard segment left
    4: 3,   # standard segment middle
    5: 4,   # standard segment right
    6: 5,   # adjacent segment right
}

# Maps Network labels -> Raw labels (for visualization/NeRF transfer later)
learning_map_inv = {
    -1: 0,
     0: 1,
     1: 2,
     2: 3,
     3: 4,
     4: 5,
     5: 6,
}

# Data pipelines required by build_dataset_single_mask
data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # Center shift the local ring for numerical stability
            dict(type="CenterShift", apply_z=True),
            
            # Voxel downsampling to save memory. 
            # If you hit OOM errors, increase grid_size (e.g., 0.05).
            dict(
                type="GridSample",
                grid_size=0.02, 
                hash_type="fnv",
                mode="train",
                # CRITICAL: We pass 'feat' so your 3-channel intensity survives voxelization
                keys=("coord", "strength", "feat", "instance", "segment"),
                return_grid_coord=True,
            ),
            
            # Bypassing the PTV3 domain/condition assertion checks
            dict(type="Add", keys_dict={"condition": "Seg2Tunnel"}),
            dict(type="Add", keys_dict={"domain": "Tunnel"}),
            
            # Convert NumPy arrays to PyTorch Tensors
            dict(type="ToTensor"),
            
            # Final packaging of the batch dictionary
            dict(
                type="Collect",
                keys=(
                    "coord", 
                    "grid_coord", 
                    "strength", 
                    "feat", 
                    "point", 
                    "masks", 
                    "condition", 
                    "domain", 
                    "mask_labels", 
                    "segment"
                ),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train", # Note: PTv3 often uses mode="train" in val for GridSample to maintain deterministic behavior
                keys=("coord", "strength", "feat", "instance", "segment"),
                return_grid_coord=True,
            ),
            dict(type="Add", keys_dict={"condition": "Seg2Tunnel"}),
            dict(type="Add", keys_dict={"domain": "Tunnel"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord", 
                    "grid_coord", 
                    "strength", 
                    "feat", 
                    "point", 
                    "masks", 
                    "condition", 
                    "domain", 
                    "mask_labels", 
                    "segment"
                ),
            ),
        ],
        test_mode=False,
    ),
)