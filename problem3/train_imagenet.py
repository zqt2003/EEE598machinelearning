"""Train a custom ResNet from scratch on an ImageNet split (prototype or full), on 1 or more GPUs.

Single GPU:   python train_imagenet.py --arch resnet36 --csv imagenet_splits/proto_split.csv --out runs/r36
Multi-GPU:    torchrun --nproc_per_node 2 train_imagenet.py --arch resnet36 ...   (DistributedDataParallel)

Writes to --out: history.csv (per epoch), results.json (final train/val/test accuracy, timing),
best.pt (best validation epoch), last.pt (for resuming after a time limit: rerun the same command).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from resnet_custom import CONFIGS, ResNet, count_layers

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=list(CONFIGS), default="resnet36")
ap.add_argument("--csv", default="imagenet_splits/proto_split.csv")
ap.add_argument("--root", default="/data/datasets/community/deeplearning/imagenet/train")
ap.add_argument("--out", default="runs/resnet36")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=128, help="batch size PER GPU")
ap.add_argument("--lr", type=float, default=0.1, help="learning rate for a total batch of 256 (scaled linearly)")
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--warmup", type=float, default=2, help="warm-up epochs")
ap.add_argument("--label-smoothing", type=float, default=0.1)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--workers", type=int, default=8, help="data-loading processes PER GPU")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-resume", action="store_true")
args = ap.parse_args()

# ---- distributed setup (works with and without torchrun) ----
cuda = torch.cuda.is_available()
world = int(os.environ.get("WORLD_SIZE", 1))
if world > 1:
    dist.init_process_group("nccl" if cuda else "gloo")
rank = dist.get_rank() if world > 1 else 0
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
if cuda:
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
main = rank == 0
torch.manual_seed(args.seed + rank)
out = Path(args.out)
if main:
    out.mkdir(parents=True, exist_ok=True)


def log(*a):
    if main:
        print(*a, flush=True)


# ---- data ----
class CSVImages(Dataset):
    def __init__(self, df, root, transform):
        self.paths, self.labels = df.path.tolist(), df.label.tolist()
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            img = im.convert("RGB")                 # some ImageNet images are grayscale or CMYK
        return self.transform(img), self.labels[i]


df = pd.read_csv(args.csv)
num_classes = int(df.label.max()) + 1
train_tf = T.Compose([T.RandomResizedCrop(args.img_size), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(int(args.img_size * 256 / 224)), T.CenterCrop(args.img_size), T.ToTensor(), T.Normalize(MEAN, STD)])
train_ds = CSVImages(df[df.split == "train"], args.root, train_tf)
val_ds = CSVImages(df[df.split == "val"], args.root, eval_tf)
test_ds = CSVImages(df[df.split == "test"], args.root, eval_tf)

loader_kw = dict(num_workers=args.workers, pin_memory=cuda, persistent_workers=args.workers > 0,
                 prefetch_factor=4 if args.workers > 0 else None)
train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler, shuffle=train_sampler is None,
                      drop_last=True, **loader_kw)


def shard(ds):  # every GPU evaluates a different 1/world of the images (no duplicates)
    return Subset(ds, range(rank, len(ds), world)) if world > 1 else ds


val_dl = DataLoader(shard(val_ds), batch_size=2 * args.bs, **loader_kw)
test_dl = DataLoader(shard(test_ds), batch_size=2 * args.bs, **loader_kw)

# ---- model (random initialization: trained from scratch) ----
model = ResNet(CONFIGS[args.arch], num_classes=num_classes).to(device, memory_format=torch.channels_last)
n_params = sum(p.numel() for p in model.parameters())
log(f"{args.arch}: blocks {CONFIGS[args.arch]}, {count_layers(model)} layers, {n_params / 1e6:.2f}M params, "
    f"{num_classes} classes | train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)} images | "
    f"{world} GPU(s) x batch {args.bs}")
ddp_model = DDP(model, device_ids=[local_rank] if cuda else None) if world > 1 else model

# SGD with momentum, no weight decay on BatchNorm/bias, linear LR scaling, warm-up + cosine decay
decay = [p for n, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
base_lr = args.lr * args.bs * world / 256
opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                      lr=base_lr, momentum=0.9, nesterov=True)
steps_per_epoch = len(train_dl)
total_steps, warmup_steps = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)


def lr_at(step):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return 0.5 * base_lr * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))


amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=cuda)


@torch.no_grad()
def evaluate(loader):
    ddp_model.eval()
    s = torch.zeros(4, device=device)            # loss sum, top-1 correct, top-5 correct, count
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = ddp_model(x)
        logits = logits.float()
        top5 = logits.topk(min(5, num_classes), 1).indices
        s += torch.stack([F.cross_entropy(logits, y, reduction="sum"), (top5[:, 0] == y).sum().float(),
                          (top5 == y[:, None]).any(1).sum().float(), torch.tensor(float(len(y)), device=device)])
    if world > 1:
        dist.all_reduce(s)
    return (s[0] / s[3]).item(), (s[1] / s[3]).item(), (s[2] / s[3]).item()


# ---- resume ----
start_epoch, history, best_val, train_time = 0, [], -1.0, 0.0
if (out / "last.pt").exists() and not args.no_resume:
    ck = torch.load(out / "last.pt", map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_epoch, history, best_val, train_time = ck["epoch"], ck["history"], ck["best_val"], ck["train_time"]
    log(f"resumed from epoch {start_epoch}")

# ---- train ----
for epoch in range(start_epoch, args.epochs):
    ddp_model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    s = torch.zeros(3, device=device)            # loss sum, correct, count
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for i, (x, y) in enumerate(train_dl):
        step = epoch * steps_per_epoch + i
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = ddp_model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        s += torch.stack([loss.detach().float() * len(y), (logits.argmax(1) == y).sum().float(),
                          torch.tensor(float(len(y)), device=device)])
    if cuda:
        torch.cuda.synchronize()
    epoch_time = time.time() - t0
    train_time += epoch_time
    if world > 1:
        dist.all_reduce(s)
    val_loss, val_top1, val_top5 = evaluate(val_dl)
    history.append(dict(epoch=epoch + 1, lr=lr_at(step), train_loss=(s[0] / s[2]).item(), train_top1=(s[1] / s[2]).item(),
                        val_loss=val_loss, val_top1=val_top1, val_top5=val_top5, epoch_time_s=epoch_time,
                        images_per_s=s[2].item() / epoch_time))
    h = history[-1]
    log(f"[{args.arch}] epoch {epoch + 1:3d}/{args.epochs}  train loss {h['train_loss']:.3f} top1 {h['train_top1']:.3f} | "
        f"val loss {val_loss:.3f} top1 {val_top1:.3f} top5 {val_top5:.3f} | {epoch_time:.0f}s, {h['images_per_s']:.0f} img/s")
    if main:
        if val_top1 > best_val:
            best_val = val_top1
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch + 1, history=history,
                        best_val=best_val, train_time=train_time), out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    if world > 1:
        dist.broadcast_object_list(best := [best_val], src=0)
        best_val = best[0]

# ---- final evaluation with the best-validation weights ----
if world > 1:
    dist.barrier()
model.load_state_dict(torch.load(out / "best.pt", map_location=device))
test_loss, test_top1, test_top5 = evaluate(test_dl)
best_ep = max(history, key=lambda h: h["val_top1"])
results = dict(arch=args.arch, blocks=list(CONFIGS[args.arch]), layers=count_layers(model), params=n_params,
               gpus=world, batch_total=args.bs * world, epochs=args.epochs, best_epoch=best_ep["epoch"],
               train_top1=best_ep["train_top1"], val_top1=best_ep["val_top1"], val_top5=best_ep["val_top5"],
               test_top1=test_top1, test_top5=test_top5, train_time_min=train_time / 60,
               avg_images_per_s=sum(h["images_per_s"] for h in history) / len(history))
if main:
    (out / "results.json").write_text(json.dumps(results, indent=2))
    log(json.dumps(results, indent=2))
if world > 1:
    dist.destroy_process_group()
