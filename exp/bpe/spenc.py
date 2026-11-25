"""Кодировщик на своём BPE (SentencePiece). Совместим по интерфейсу с swenc.Subword."""
import os

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
BPE_MODEL = os.path.join(HERE, "bpe.model")


class SPSubword:
    def __init__(self, model_path=BPE_MODEL):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.size = self.sp.get_piece_size()

    def __len__(self):
        return self.size + 1          # +1: blank

    def encode(self, text):
        ids = self.sp.encode(text.lower(), out_type=int)
        return [i + 1 for i in ids]   # 0 = blank

    def decode(self, ids):
        out, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                out.append(i - 1)
            prev = i
        return self.sp.decode(out).strip()
