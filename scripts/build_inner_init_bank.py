#!/usr/bin/env python3
"""
build_inner_init_bank.py

Train a bank of supervised affinity models from random initialization and save
their final checkpoints. These checkpoints can then be used as warm-start
states for inner models during DataRater meta-training.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_trainer import train_baseline
from data_utils import prepare_data


LOG = logging.getLogger("build_inner_init_bank")


def parse_args():
    p = argparse.ArgumentParser(description="Build a checkpoint bank for inner-model warm starts")
    p.add_argument("--dataset", type=str, default="Bindwell/PPBA")
    p.add_argument("--data_mode", type=str, default="all", choices=["combined_train", "all"])
    p.add_argument("--exclude_sources", type=str, default="",
                   help="Comma-separated source names to exclude before splitting/tokenization")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42, help="Data split seed")
    p.add_argument("--bank_base_seed", type=int, default=1000, help="Base seed for random bank members")
    p.add_argument("--bank_epochs", type=str, required=True,
                   help="Comma-separated epoch list, e.g. 5,5,6,6,7,7,8,8")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--esm_attn_implementation", type=str, default="auto",
                   choices=["auto", "sdpa", "flash_attention_2", "eager"])
    p.add_argument("--esm_torch_dtype", type=str, default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def setup_logging(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "build.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_epoch_list(raw: str):
    epochs = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not epochs:
        raise ValueError("bank_epochs must contain at least one integer.")
    if any(epoch <= 0 for epoch in epochs):
        raise ValueError(f"bank_epochs must be positive, got: {epochs}")
    return epochs


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"inner_init_bank_{timestamp}")
    setup_logging(run_dir)
    LOG.info("Building inner-init checkpoint bank -> %s", run_dir)
    LOG.info("Args: %s", json.dumps(vars(args), indent=2))

    bank_epochs = parse_epoch_list(args.bank_epochs)
    with open(os.path.join(run_dir, "bank_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)

    train_loader, val_loader, _, _, _, _ = prepare_data(
        dataset_name=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        mode=args.data_mode,
        exclude_sources=[part.strip() for part in str(args.exclude_sources).split(",") if part.strip()],
    )

    members = []
    for member_idx, target_epochs in enumerate(bank_epochs):
        member_seed = int(args.bank_base_seed + member_idx)
        member_id = f"member_{member_idx:02d}_ep{target_epochs}_seed{member_seed}"
        member_dir = os.path.join(run_dir, member_id)
        final_ckpt = os.path.join(member_dir, "final.pt")
        if os.path.exists(final_ckpt):
            LOG.info("Skipping existing member %s", member_id)
            member_results = json.load(open(os.path.join(member_dir, "member_results.json")))
            members.append(member_results)
            continue

        os.makedirs(member_dir, exist_ok=True)
        set_seed(member_seed)
        LOG.info("[%d/%d] Training %s", member_idx + 1, len(bank_epochs), member_id)
        result = train_baseline(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=target_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            save_dir=member_dir,
            tag=f"inner_bank_{member_idx:02d}",
            device=device,
            attn_implementation=args.esm_attn_implementation,
            esm_torch_dtype=args.esm_torch_dtype,
        )
        torch.save(result["model"].state_dict(), final_ckpt)
        member_results = {
            "member_id": member_id,
            "index": int(member_idx),
            "epoch": int(target_epochs),
            "seed": int(member_seed),
            "final_checkpoint": final_ckpt,
            "best_metrics": result["best_metrics"],
            "history": result["history"],
        }
        with open(os.path.join(member_dir, "member_results.json"), "w") as f:
            json.dump(member_results, f, indent=2)
        members.append(member_results)

    manifest = {
        "run_dir": run_dir,
        "resolved_config": {
            "dataset": args.dataset,
            "data_mode": args.data_mode,
            "exclude_sources": [part.strip() for part in str(args.exclude_sources).split(",") if part.strip()],
            "train_ratio": float(args.train_ratio),
            "seed": int(args.seed),
            "bank_base_seed": int(args.bank_base_seed),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "esm_attn_implementation": str(args.esm_attn_implementation),
            "esm_torch_dtype": str(args.esm_torch_dtype),
        },
        "members": members,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    LOG.info("Inner-init bank complete -> %s", os.path.join(run_dir, "manifest.json"))


if __name__ == "__main__":
    main()
