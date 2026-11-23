"""CTC с субсловами Яндекса (выход 1000) — обучение с нуля и оценка.

Данные берём из основного проекта (C:\\ZESP_asr\\train), но ничего там не меняем.

    python swtrain.py --epochs 20 --batch 16 --lr 3e-4
"""
import argparse
import io
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"C:\ZESP_asr\train"
sys.path.insert(0, HERE)
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import paths
import datalib
from train_ctc import CTCModel, edit_distance, spec_augment
from swenc import Subword

CKPT = os.path.join(HERE, "model_subword.pt")


def read_meta(path, limit=0):
    rows = []
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 2 and p[0] != "id" and p[1].strip():
            rows.append((p[0], p[1]))
        if limit and len(rows) >= limit:
            break
    return rows


class DS(Dataset):
    def __init__(self, items, sw, train=True):
        self.items = items
        self.sw = sw
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, text = self.items[i]
        m = torch.load(paths.mel_path(idx), weights_only=True).float()
        if self.train:
            m = m.clone()
            m = spec_augment(m)
        y = torch.tensor(self.sw.encode(text), dtype=torch.long)
        return m, y


def collate(batch):
    ms, ys = zip(*batch)
    lens = torch.tensor([m.shape[0] for m in ms])
    ylens = torch.tensor([len(y) for y in ys])
    m = torch.zeros(len(ms), int(lens.max()), ms[0].shape[1])
    for i, x in enumerate(ms):
        m[i, :x.shape[0]] = x
    y = torch.cat(ys) if ys else torch.zeros(0, dtype=torch.long)
    return m, lens, y, ylens


@torch.no_grad()
def evaluate(model, items, sw, dev, amp=True):
    model.eval()
    num = den = 0
    for s in range(0, len(items), 16):
        chunk = items[s:s + 16]
        ms = [torch.load(paths.mel_path(i), weights_only=True).float() for i, _ in chunk]
        lens = torch.tensor([m.shape[0] for m in ms])
        m = torch.zeros(len(ms), int(lens.max()), ms[0].shape[1])
        for k, x in enumerate(ms):
            m[k, :x.shape[0]] = x
        m, lens = m.to(dev), lens.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _ = model(m, lens)
        ids = logits.float().argmax(-1).cpu()
        for k, (_, ref) in enumerate(chunk):
            hyp = sw.decode(ids[k].tolist())
            num += edit_distance(ref.lower(), hyp)
            den += max(1, len(ref))
    model.train()
    return num / max(1, den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tts-limit", type=int, default=0)
    ap.add_argument("--golos", type=int, default=0, help="сколько фраз Golos farfield добавить")
    ap.add_argument("--live-weight", type=float, default=1.0, help="перекос живых фраз")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)

    sw = Subword()
    print("субслов в словаре: %d, выход модели: %d" % (len(sw.tokens), len(sw)), flush=True)

    tts = read_meta(paths.META, args.tts_limit)
    live = read_meta(paths.LIVE_META)
    nov = read_meta(os.path.join(paths.DATA, "live_nov23_v2_train.csv"))

    # Человеческий val .128 фиксируем по старому meta (.bak), как в основных замерах;
    # из train эти id исключаем, иначе оценка завышена.
    bak = paths.LIVE_META + ".bak"
    val_ids = {i for i, _ in read_meta(bak) if datalib.item_index(i) % 5 == 0}
    live_val = [i for i in live if i[0] in val_ids]
    live_train = [i for i in live if i[0] not in val_ids]
    nov_val = read_meta(os.path.join(paths.DATA, "live_nov23_v2_val.csv"))

    train_items = [i for i in tts + live_train + nov
                   if os.path.exists(paths.mel_path(i[0]))]
    if args.golos:
        golos = [i for i in read_meta(paths.FARFIELD_META, args.golos)
                 if os.path.exists(paths.mel_path(i[0]))]
        train_items += golos
        print("golos: %d фраз" % len(golos), flush=True)
    train_items = [(i, t) for i, t in train_items if sw.encode(t)]
    print("train %d, val .128 %d, val .151 %d"
          % (len(train_items), len(live_val), len(nov_val)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CTCModel(len(sw)).to(dev)
    print("params %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    dl = DataLoader(DS(train_items, sw), batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)
    if args.live_weight != 1.0:
        from torch.utils.data import WeightedRandomSampler
        w = [args.live_weight if i[0].startswith("l") else 1.0 for i in train_items]
        n_live = sum(1 for i in train_items if i[0].startswith("l"))
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        dl = DataLoader(DS(train_items, sw), batch_size=args.batch, sampler=sampler,
                        collate_fn=collate, num_workers=0, drop_last=True)
        print("перекос живых x%.1f (живых %d из %d)"
              % (args.live_weight, n_live, len(train_items)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    amp = args.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = 9e9
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for m, lens, y, ylens in dl:
            m, lens, y, ylens = m.to(dev), lens.to(dev), y.to(dev), ylens.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, _ = model(m, lens)
            out_lens = (lens + 3) // 4
            loss = ctc(logits.float().transpose(0, 1), y, out_lens, ylens)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            n += 1
        sched.step()
        c128 = evaluate(model, live_val, sw, dev, amp)
        c151 = evaluate(model, nov_val, sw, dev, amp)
        print("ep %d loss %.3f  CER .128 %.4f  .151 %.4f  %.0fs"
              % (ep, tot / max(1, n), c128, c151, time.time() - t0), flush=True)
        score = c151 + c128
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "tokens": sw.tokens}, CKPT)
            print("  saved -> %s" % CKPT, flush=True)


if __name__ == "__main__":
    main()
