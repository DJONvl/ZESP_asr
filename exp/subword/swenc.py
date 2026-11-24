"""Словарь Яндекса: загрузка, кодирование текста в субслова, обратная сборка."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(HERE, "subword_dict.txt")
BLANK = "<blank>"
UNK = "<unk>"


class Subword:
    def __init__(self, path=DICT_PATH):
        toks = []
        for line in io.open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            toks.append(line.rsplit(" ", 1)[0])
        self.tokens = toks
        self.vocab = {t: i + 1 for i, t in enumerate(toks)}   # 0 = blank
        self.inv = {i + 1: t for i, t in enumerate(toks)}
        self.maxlen = max(len(t) for t in toks)

    def __len__(self):
        return len(self.tokens) + 1

    def encode(self, text):
        ids = []
        for word in text.split():
            s = "\u2581" + word.lower()
            i = 0
            while i < len(s):
                for ln in range(min(self.maxlen, len(s) - i), 0, -1):
                    piece = s[i:i + ln]
                    k = self.vocab.get(piece)
                    if k is not None:
                        ids.append(k)
                        i += ln
                        break
                else:
                    i += 1          # символ без токена — пропускаем (0.00% случаев)
        return ids

    def decode(self, ids):
        """CTC greedy: схлопываем повторы и убираем blank (id 0)."""
        out, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                out.append(self.inv.get(i, ""))
            prev = i
        return "".join(out).replace("\u2581", " ").strip()
