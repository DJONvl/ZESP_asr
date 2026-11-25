"""Оценка BPE-модели на 244 фразах: CER/WER и «команд верно»."""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\ZESP_asr\train")
sys.path.insert(0, HERE)
os.chdir(r"C:\ZESP_asr\train")

import torch

import paths
import datalib
from train_ctc import CTCModel
from eval_wer import load_items, score
from eval_final import best_match
from spenc import SPSubword

OUT = os.path.join(HERE, "logs", "eval_bpe.txt")


def eval_bpe(ckpt, items):
    sw = SPSubword()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CTCModel(len(sw))
    m.load_state_dict(ck["model"])
    m.eval()
    res = []
    for s in range(0, len(items), 16):
        chunk = items[s:s + 16]
        ms = [torch.load(paths.mel_path(i), weights_only=True).float() for i, _ in chunk]
        L = torch.tensor([x.shape[0] for x in ms])
        M = torch.zeros(len(ms), int(L.max()), ms[0].shape[1])
        for k, x in enumerate(ms):
            M[k, :x.shape[0]] = x
        with torch.no_grad():
            lg, _ = m(M, L)
        ids = lg.argmax(-1)
        for k, (_, ref) in enumerate(chunk):
            res.append((ref, sw.decode(ids[k].tolist())))
    return res


def main():
    old = [i for i in load_items(paths.LIVE_META + ".bak")
           if datalib.item_index(i[0]) % 5 == 0]
    nov = load_items(os.path.join(paths.DATA, "live_nov23_v2_val.csv"))
    items = old + nov
    templates = sorted({t for _, t in items})

    runs = [
        ("domain7 (буквы)", r"C:\ZESP_asr\train\models\model_domain7.pt", "char"),
        ("subword Яндекса", r"C:\ZESP_asr\exp\subword\model_subword_aug2.pt", "sub"),
        ("свой BPE", os.path.join(HERE, "model_bpe_aug.pt"), "bpe"),
    ]
    lines = ["фраз: %d" % len(items), "",
             "%-20s %8s %8s %10s" % ("модель", "CER", "WER", "команд")]
    for name, path, kind in runs:
        if not os.path.exists(path):
            continue
        if kind == "char":
            from eval_final import eval_char
            pairs = eval_char(path, items)
        elif kind == "sub":
            from eval_final import eval_subword
            pairs = eval_subword(path, items)
        else:
            pairs = eval_bpe(path, items)
        c, w = score(pairs)
        ok = sum(1 for ref, hyp in pairs if best_match(hyp, templates) == ref)
        lines.append("%-20s %8.4f %8.4f %9.1f%%" % (name, c, w, 100.0 * ok / len(pairs)))
        print(lines[-1], flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = "\n".join(lines)
    open(OUT, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
