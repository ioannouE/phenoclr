#!/usr/bin/env python3
"""
Extract DinoV3 backbone weights from a SimCLR Lightning checkpoint.
- Loads your trained SimCLR model (DinoV3 backbone) via Lightning.
- Extracts only the DinoV3 backbone state_dict (excluding projection head).
- Optionally prints key names and validates basic DinoV3 compatibility.
- Saves the backbone weights to a .pth file for later reuse.

Usage:
  python simclr/scripts/extract_dinov3_from_simclr.py \
      --ckpt /path/to/checkpoint.ckpt \
      --config /path/to/config.yaml \
      --out /path/to/dinov3_backbone.pth \
      [--print-keys] [--verify-dinov3]

Note:
- This saves raw DinoV3 keys compatible with torch.hub DinoV3 models.
- The extracted weights can be used with torch.hub.load() for DinoV3 models.
"""

import argparse
import json
import os
from typing import Dict, Any

import torch
import yaml

# Import your Lightning SimCLR model with DinoV3 backbone
from simclr_birdcolour_kornia_dino import SimCLRModel


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Build the minimal model_config expected by SimCLRModel.load_from_checkpoint
    model_cfg = {
        "scheduler": cfg["model"]["scheduler"],
        "optimizer": cfg["model"]["optimizer"],
        "criterion": cfg["model"]["criterion"],
        "dinov3": cfg["model"]["dinov3"],
    }
    return model_cfg


def _gather_meta(backbone: torch.nn.Module) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    # Try to determine DinoV3 architecture from model structure
    try:
        # Check for common DinoV3 attributes
        if hasattr(backbone, 'blocks'):
            meta["num_blocks"] = len(backbone.blocks)
            
            # Get embedding dimension from first block
            if len(backbone.blocks) > 0:
                first_block = backbone.blocks[0]
                if hasattr(first_block, 'norm1') and hasattr(first_block.norm1, 'weight'):
                    meta["embed_dim"] = int(first_block.norm1.weight.shape[0])
                elif hasattr(first_block, 'attn') and hasattr(first_block.attn, 'qkv'):
                    # Infer from attention layer
                    qkv_weight = first_block.attn.qkv.weight
                    meta["embed_dim"] = int(qkv_weight.shape[1])
        
        # Check patch embedding
        if hasattr(backbone, 'patch_embed'):
            patch_embed = backbone.patch_embed
            if hasattr(patch_embed, 'proj'):
                proj_weight = patch_embed.proj.weight  # (embed_dim, in_chans, patch_h, patch_w)
                meta["in_chans"] = int(proj_weight.shape[1])
                meta["patch_size"] = int(proj_weight.shape[-1])
                meta["embed_dim_from_patch"] = int(proj_weight.shape[0])
        
        # Check positional embedding
        if hasattr(backbone, 'pos_embed'):
            pos_embed = backbone.pos_embed  # (1, num_patches + 1, embed_dim)
            meta["pos_embed_shape"] = list(pos_embed.shape)
            seq_len = int(pos_embed.shape[1])
            meta["seq_len"] = seq_len
            meta["has_cls_token"] = True  # DinoV3 includes CLS token
            meta["num_patches"] = seq_len - 1
            
        # Check class token
        if hasattr(backbone, 'cls_token'):
            cls_token = backbone.cls_token
            meta["cls_token_shape"] = list(cls_token.shape)
            
        # Try to determine architecture variant
        embed_dim = meta.get("embed_dim", meta.get("embed_dim_from_patch", 0))
        if embed_dim >= 4000:
            meta["likely_arch"] = "dinov3_vit7b16"
        elif embed_dim >= 1024:
            meta["likely_arch"] = "dinov3_vitl16"
        elif embed_dim >= 768:
            meta["likely_arch"] = "dinov3_vitb16"
        elif embed_dim >= 384:
            meta["likely_arch"] = "dinov3_vits16"
        else:
            meta["likely_arch"] = "unknown"
            
    except Exception as e:
        meta["error_gathering_meta"] = str(e)

    return meta


def _verify_for_dinov3(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small report on basic DinoV3 compatibility expectations."""
    report = {}
    
    embed_dim = meta.get("embed_dim", meta.get("embed_dim_from_patch", 0))
    
    # Check common DinoV3 configurations
    report["has_cls_token"] = meta.get("has_cls_token", False)
    report["patch_size_16"] = (meta.get("patch_size") == 16)
    report["in_chans_3"] = (meta.get("in_chans") == 3)
    report["embed_dim"] = embed_dim
    report["likely_architecture"] = meta.get("likely_arch", "unknown")
    
    # Check if dimensions match known DinoV3 variants
    known_dims = {384: "vits16", 768: "vitb16", 1024: "vitl16", 4096: "vit7b16"}
    report["matches_known_dinov3_variant"] = embed_dim in known_dims
    
    if embed_dim in known_dims:
        report["variant"] = known_dims[embed_dim]
    
    # Training image size often 224 -> seq_len = 1 + 14*14 = 197
    if "seq_len" in meta and "patch_size" in meta:
        if meta["seq_len"] > 1:
            grid_size = int((meta["seq_len"] - 1) ** 0.5)
            report["train_grid_size"] = grid_size
            report["implied_image_size"] = grid_size * meta.get("patch_size", 16)
    
    return report


def extract_and_save(ckpt: str, config_path: str, out_path: str, print_keys: bool, verify_dinov3: bool):
    # Load model
    model_config = _load_config(config_path)
    print("Loading SimCLR Lightning checkpoint:", ckpt)
    model = SimCLRModel.load_from_checkpoint(ckpt, config=model_config, map_location="cpu")
    model.eval()

    # Extract DinoV3 backbone weights only
    dinov3_state = model.backbone.state_dict()

    # Optional introspection
    meta = _gather_meta(model.backbone)
    print("DinoV3 Backbone meta:")
    print(json.dumps(meta, indent=2))

    if verify_dinov3:
        report = _verify_for_dinov3(meta)
        print("\nDinoV3 compatibility checks:")
        print(json.dumps(report, indent=2))

    if print_keys:
        print("\nState dict keys (DinoV3 backbone):")
        for k in sorted(dinov3_state.keys()):
            print(" ", k)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Save the backbone state dict with metadata
    save_dict = {
        'state_dict': dinov3_state,
        'meta': meta,
        'architecture': meta.get('likely_arch', 'unknown'),
        'embed_dim': meta.get('embed_dim', meta.get('embed_dim_from_patch', 0))
    }
    
    print(f"Saving DinoV3 backbone to: {out_path}")
    torch.save(save_dict, out_path)
    
    print(f" DinoV3 backbone extracted successfully!")
    print(f"  Architecture: {meta.get('architecture', 'unknown')}")
    print(f"  Embedding dim: {meta.get('embed_dim', 'unknown')}")
    print(f"  Patch size: {meta.get('patch_size', 'unknown')}")
    print(f"  Num blocks: {meta.get('num_blocks', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Extract DinoV3 backbone from SimCLR checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to SimCLR Lightning checkpoint (.ckpt)")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--out", required=True, help="Output path for DinoV3 backbone (.pth)")
    parser.add_argument("--print-keys", action="store_true", help="Print all state dict keys")
    parser.add_argument("--verify-dinov3", action="store_true", help="Run DinoV3 compatibility checks")
    
    args = parser.parse_args()
    
    extract_and_save(
        ckpt=args.ckpt,
        config_path=args.config,
        out_path=args.out,
        print_keys=args.print_keys,
        verify_dinov3=args.verify_dinov3
    )


if __name__ == "__main__":
    main()
