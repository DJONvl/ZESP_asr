"""Свой BPE на нашем корпусе (вместо словаря Яндекса).

    python train_bpe.py

Собирает тексты из TTS + живых записей + Golos и обучает SentencePiece BPE на
999 токенов. Результат: bpe.model, bpe.vocab (по токену на строку).
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\ZESP_asr\train")
os.chdir(r"C:\ZESP_asr\train")

import paths

CORPUS = os.path.join(HERE, "corpus.txt")
MODEL_PREFIX = os.path.join(HERE, "bpe")
VOCAB_OUT = os.path.join(HERE, "bpe.vocab")


def read_texts(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 2 and p[0] != "id" and p[1].strip():
            out.append(p[1].strip().lower())
    return out


def main():
    texts = []
    for path in (paths.META, paths.LIVE_META,
                 os.path.join(paths.DATA, "live_nov23_v2.csv"),
                 paths.FARFIELD_META):
        part = read_texts(path)
        print("%-40s %d строк" % (os.path.basename(path), len(part)), flush=True)
        texts += part

    # дубликаты не нужны: BPE учит частоты, но повторы искажают баланс доменов
    seen = set()
    uniq = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    print("всего %d, уникальных %d" % (len(texts), len(uniq)), flush=True)

    with io.open(CORPUS, "w", encoding="utf-8") as f:
        for t in uniq:
            f.write(t + "\n")

    import sentencepiece as spm

    # доменные слова из словаря команд ZESP — фиксируем целыми токенами
    import json
    lex_path = os.path.join(r"C:\ZESP_asr\train", "lexicon.json")
    user_words = []
    if os.path.exists(lex_path):
        j = json.load(io.open(lex_path, encoding="utf-8"))
        seen_w = set()
        for k, v in j.items():
            if not isinstance(v, list):
                continue
            for x in v:
                if isinstance(x, str):
                    w = x.lower().strip()
                    if w and " " not in w and w not in seen_w:
                        seen_w.add(w)
                        user_words.append(w)
        print("доменных слов: %d" % len(user_words), flush=True)

    spm.SentencePieceTrainer.train(
        input=CORPUS,
        model_prefix=MODEL_PREFIX,
        vocab_size=999,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc",
        user_defined_symbols=user_words,
        num_threads=8,
    )

    sp = spm.SentencePieceProcessor(model_file=MODEL_PREFIX + ".model")
    with io.open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(sp.id_to_piece(i) + "\n")
    print("токенов: %d -> %s" % (sp.get_piece_size(), VOCAB_OUT))

    for w in ("зеркало", "включи зеркало", "громкость", "приставка", "подсветка",
              "телевизор", "свет"):
        pieces = sp.encode(w, out_type=str)
        print("  %-20s -> %s" % (w, pieces))


if __name__ == "__main__":
    main()
