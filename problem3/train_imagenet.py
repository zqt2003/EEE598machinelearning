"""Train a custom ResNet from scratch on an ImageNet split (prototype or full) on NVIDIA GPUs or Intel Gaudi (HPU).

NVIDIA GPU, 1 device:   python train_imagenet.py --arch resnet36 --act arcgate --csv imagenet_splits/full_split.csv
NVIDIA GPU, N devices:  torchrun --nproc_per_node N train_imagenet.py ...            (DDP over NCCL)
Intel Gaudi, N cards:   torchrun --nproc_per_node N train_imagenet.py --device hpu ...  (DDP over HCCL)

Writes to --out: history.csv (per epoch), results.json (accuracy, timing, hyperparameters, inference latency),
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

from resnet_custom import ACTIVATIONS, CONFIGS, ResNet, count_layers

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

ap = argparse.ArgumentParser()
ap.add_argument("--device", choices=["auto", "cuda", "hpu", "cpu"], default="auto", help="hpu = Intel Gaudi")
ap.add_argument("--arch", choices=list(CONFIGS), default="resnet36")
ap.add_argument("--act", choices=list(ACTIVATIONS), default="relu", help="activation function")
ap.add_argument("--head", choices=["linear", "kan"], default="linear", help="classifier head")
ap.add_argument("--csv", default="imagenet_splits/proto_split.csv")
ap.add_argument("--root", default="/data/datasets/community/deeplearning/imagenet/train")
ap.add_argument("--out", default="runs/resnet36")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=128, help="batch size PER device")
ap.add_argument("--lr", type=float, default=0.1, help="learning rate for a total batch of 256 (scaled linearly)")
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--warmup", type=float, default=2, help="warm-up epochs")
ap.add_argument("--label-smoothing", type=float, default=0.1)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--workers", type=int, default=8, help="data-loading processes PER device")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-resume", action="store_true")
ap.add_argument("--latency-iters", type=int, default=100, help="timed forward passes per batch size (0 = skip)")
args = ap.parse_args()

# ---- device + distributed setup (works with and without torchrun) ----
dev_type = args.device
if dev_type == "auto":
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
if dev_type == "hpu":
    import habana_frameworks.torch as htorch                 # Intel Gaudi PyTorch bridge
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl  # noqa: F401  (registers the 'hccl' backend)

world = int(os.environ.get("WORLD_SIZE", 1))
if world > 1:
    dist.init_process_group({"cuda": "nccl", "hpu": "hccl"}.get(dev_type, "gloo"))
rank = dist.get_rank() if world > 1 else 0
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if dev_type == "cuda":
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device(dev_type)                          # 'hpu': each process gets its own Gaudi card
main = rank == 0
torch.manual_seed(args.seed + rank)
out = Path(args.out)
if main:
    out.mkdir(parents=True, exist_ok=True)

mem_fmt = torch.channels_last if dev_type == "cuda" else torch.contiguous_format
amp = dict(device_type=dev_type, dtype=torch.bfloat16, enabled=dev_type in ("cuda", "hpu"))


def sync():
    """Wait until the accelerator has finished its queued work (for correct timing)."""
    if dev_type == "cuda":
        torch.cuda.synchronize()
    elif dev_type == "hpu":
        htorch.hpu.synchronize()


def mark_step():
    """Gaudi lazy mode: execute the accumulated graph. No-op on other devices."""
    if dev_type == "hpu":
        htcore.mark_step()


def cpu_state(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def device_name():
    try:
        if dev_type == "cuda":
            return torch.cuda.get_device_name(device)
        if dev_type == "hpu":
            return htorch.hpu.get_device_name()
    except Exception:
        pass
    return dev_type


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

loader_kw = dict(num_workers=args.workers, pin_memory=dev_type == "cuda", persistent_workers=args.workers > 0,
                 prefetch_factor=4 if args.workers > 0 else None)
train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler, shuffle=train_sampler is None,
                      drop_last=True, **loader_kw)


def shard(ds):  # every device evaluates a different 1/world of the images (no duplicates)
    return Subset(ds, range(rank, len(ds), world)) if world > 1 else ds


val_dl = DataLoader(shard(val_ds), batch_size=2 * args.bs, **loader_kw)
test_dl = DataLoader(shard(test_ds), batch_size=2 * args.bs, **loader_kw)

# ---- model (random initialization: trained from scratch) ----
model = ResNet(CONFIGS[args.arch], num_classes=num_classes, act=args.act, head=args.head).to(device, memory_format=mem_fmt)
n_params = sum(p.numel() for p in model.parameters())
log(f"{args.arch} ({args.act}, {args.head} head): blocks {CONFIGS[args.arch]}, {count_layers(model)} layers, {n_params / 1e6:.2f}M params, "
    f"{num_classes} classes | train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)} images | "
    f"{world} x {device_name()} x batch {args.bs}")
if world > 1:
    if dev_type == "hpu":   # settings recommended for Gaudi
        ddp_model = DDP(model, bucket_cap_mb=100, broadcast_buffers=False, gradient_as_bucket_view=True)
    else:
        ddp_model = DDP(model, device_ids=[local_rank] if dev_type == "cuda" else None)
else:
    ddp_model = model

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


@torch.no_grad()
def evaluate(loader):
    ddp_model.eval()
    s = torch.zeros(4, device=device)            # loss sum, top-1 correct, top-5 correct, count
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=mem_fmt)
        y = y.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = ddp_model(x)
        logits = logits.float()
        top5 = logits.topk(min(5, num_classes), 1).indices
        s += torch.stack([F.cross_entropy(logits, y, reduction="sum"), (top5[:, 0] == y).sum().float(),
                          (top5 == y[:, None]).any(1).sum().float(), torch.tensor(float(len(y)), device=device)])
        mark_step()
    if world > 1:
        dist.all_reduce(s)
    s = s.cpu()
    return (s[0] / s[3]).item(), (s[1] / s[3]).item(), (s[2] / s[3]).item()


@torch.no_grad()
def measure_latency(net, batch_sizes=(1, 64), warmup=20):
    """Inference latency of the trained model on this device (bf16), excluding data loading."""
    net.eval()
    res = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=device).contiguous(memory_format=mem_fmt)
        times = []
        for i in range(warmup + args.latency_iters):
            sync()
            t = time.perf_counter()
            with torch.autocast(**amp):
                net(x)
            mark_step()
            sync()
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
        t = torch.tensor(times)
        res[f"batch_{bs}"] = dict(mean_ms=round(t.mean().item(), 3), p50_ms=round(t.median().item(), 3),
                                  p90_ms=round(t.quantile(0.9).item(), 3),
                                  images_per_s=round(bs * 1000 / t.mean().item(), 1))
    return res


# ---- resume ----
start_epoch, history, best_val, train_time = 0, [], -1.0, 0.0
if (out / "last.pt").exists() and not args.no_resume:
    ck = torch.load(out / "last.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_epoch, history, best_val, train_time = ck["epoch"], ck["history"], ck["best_val"], ck["train_time"]
    log(f"resumed from epoch {start_epoch}")

# ---- train ----
wall_start = time.time()
for epoch in range(start_epoch, args.epochs):
    ddp_model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    s = torch.zeros(3, device=device)            # loss sum, correct, count
    sync()
    t0 = time.time()
    for i, (x, y) in enumerate(train_dl):
        step = epoch * steps_per_epoch + i
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x = x.to(device, non_blocking=True, memory_format=mem_fmt)
        y = y.to(device, non_blocking=True)
        with torch.autocast(**amp):
            logits = ddp_model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mark_step()
        opt.step()
        mark_step()
        s += torch.stack([loss.detach().float() * len(y), (logits.argmax(1) == y).sum().float(),
                          torch.tensor(float(len(y)), device=device)])
        if main and i % 500 == 0 and i > 0:    # progress inside long (full-dataset) epochs
            log(f"  epoch {epoch + 1} step {i}/{steps_per_epoch}  ({(time.time() - t0) / 60:.1f} min)")
    sync()
    epoch_time = time.time() - t0
    train_time += epoch_time
    if world > 1:
        dist.all_reduce(s)
    s = s.cpu()
    t_eval = time.time()
    val_loss, val_top1, val_top5 = evaluate(val_dl)
    history.append(dict(epoch=epoch + 1, lr=lr_at(step), train_loss=(s[0] / s[2]).item(), train_top1=(s[1] / s[2]).item(),
                        val_loss=val_loss, val_top1=val_top1, val_top5=val_top5, epoch_time_s=epoch_time,
                        val_time_s=time.time() - t_eval, images_per_s=s[2].item() / epoch_time))
    h = history[-1]
    log(f"[{args.arch}/{args.act}/{args.head}] epoch {epoch + 1:3d}/{args.epochs}  train loss {h['train_loss']:.3f} top1 {h['train_top1']:.3f} | "
        f"val loss {val_loss:.3f} top1 {val_top1:.3f} top5 {val_top5:.3f} | {epoch_time:.0f}s, {h['images_per_s']:.0f} img/s")
    if main:
        if val_top1 > best_val:
            best_val = val_top1
            torch.save(cpu_state(model), out / "best.pt")
        torch.save(dict(model=cpu_state(model), opt=opt.state_dict(), epoch=epoch + 1, history=history,
                        best_val=best_val, train_time=train_time), out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    if world > 1:
        dist.broadcast_object_list(best := [best_val], src=0)
        best_val = best[0]

# ---- final evaluation with the best-validation weights ----
if world > 1:
    dist.barrier()
model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
t_test = time.time()
test_loss, test_top1, test_top5 = evaluate(test_dl)
test_time = time.time() - t_test
latency = measure_latency(model) if (main and args.latency_iters > 0) else {}
best_ep = max(history, key=lambda h: h["val_top1"])
results = dict(
    arch=args.arch, act=args.act, head=args.head, blocks=list(CONFIGS[args.arch]), layers=count_layers(model), params=n_params,
    device=device_name(), devices=world, gpus=world, batch_total=args.bs * world, epochs=args.epochs, best_epoch=best_ep["epoch"],
    train_top1=best_ep["train_top1"], val_top1=best_ep["val_top1"], val_top5=best_ep["val_top5"],
    test_top1=test_top1, test_top5=test_top5, test_images=len(test_ds), test_eval_time_s=test_time,
    train_time_min=train_time / 60, avg_images_per_s=sum(h["images_per_s"] for h in history) / len(history),
    wall_time_this_run_min=(time.time() - wall_start) / 60,
    hyperparameters=dict(optimizer="SGD, momentum 0.9, Nesterov", base_lr=base_lr, lr_per_256=args.lr,
                         weight_decay=args.wd, warmup_epochs=args.warmup, schedule="linear warm-up + cosine decay",
                         label_smoothing=args.label_smoothing, batch_per_device=args.bs, global_batch=args.bs * world,
                         img_size=args.img_size, precision="bfloat16 autocast", augmentation="RandomResizedCrop + HorizontalFlip",
                         seed=args.seed),
    inference_latency=latency)
if main:
    (out / "results.json").write_text(json.dumps(results, indent=2))
    log(json.dumps(results, indent=2))
if world > 1:
    dist.destroy_process_group()
