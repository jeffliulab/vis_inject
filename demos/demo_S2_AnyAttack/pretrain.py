"""
AnyAttack self-supervised pre-training on LAION-Art.

Trains the Decoder network to generate adversarial noise that shifts CLIP
embeddings of arbitrary images toward target embeddings, using InfoNCE
contrastive loss with K-augmentation.

Usage:
    python pretrain.py
    python pretrain.py --tar-dir /path/to/laion-art/webdataset --epochs 5
    python pretrain.py --checkpoint checkpoints/pre-trained.pt  # resume
"""

import argparse
import os
import sys
import time

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.dirname(__file__))
from config import PRETRAIN_CONFIG, ATTACK_CONFIG
from models import CLIPEncoder, Decoder
from losses import DynamicInfoNCELoss
from dataset import make_laion_dataloader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)
    decoder = Decoder(embed_dim=ATTACK_CONFIG["embed_dim"]).to(device)
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()
    criterion = DynamicInfoNCELoss()

    start_epoch = 0
    global_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    train_loader = make_laion_dataloader(args.tar_dir, args.batch_size,
                                         max_shards=args.max_shards)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        count = 0
        t0 = time.time()

        for batch_idx, (images, _texts) in enumerate(train_loader):
            images = images.to(device)

            with torch.no_grad():
                e1 = clip_encoder.encode_img(images)

            with autocast():
                noise = decoder(e1)
                noise = torch.clamp(noise, -args.eps, args.eps)

                # K-augmentation: add noise to K shuffled copies, average embeddings
                adv_embeddings = []
                for _ in range(args.chunk):
                    shuffled = images[torch.randperm(images.size(0))]
                    adv_img = torch.clamp(noise + shuffled, 0, 1)
                    adv_embeddings.append(clip_encoder.encode_img_with_grad(adv_img))
                e2 = torch.stack(adv_embeddings).mean(dim=0)

                loss = criterion(e1, e2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            count += 1
            global_step += 1
            scheduler.step()

            if batch_idx <= 10000:
                criterion.update_temperature(batch_idx)

            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = total_loss / count
                elapsed = time.time() - t0
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {avg_loss:.6f}, LR: {lr:.2e}, "
                      f"Temp: {criterion.current_temp:.4f}, "
                      f"Time: {elapsed:.0f}s")

            if batch_idx % args.checkpoint_every == 0 and batch_idx > 0:
                save_path = os.path.join(args.checkpoint_dir, "pre-trained.pt")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

        # End-of-epoch save
        save_path = os.path.join(args.checkpoint_dir, "pre-trained.pt")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, save_path)
        print(f"Epoch {epoch} done. Avg loss: {total_loss / max(count, 1):.6f}. "
              f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnyAttack self-supervised pre-training")
    cfg = PRETRAIN_CONFIG
    parser.add_argument("--tar-dir", type=str, default=cfg["tar_dir"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--chunk", type=int, default=cfg["chunk"])
    parser.add_argument("--eps", type=float, default=ATTACK_CONFIG["eps"])
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Only use the first N tar shards (default: all)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=cfg["checkpoint_dir"])
    parser.add_argument("--checkpoint-every", type=int, default=cfg["checkpoint_every"])
    main(parser.parse_args())
