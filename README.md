# ZESP_asr

Файлы офлайн-распознавания речи (ASR) и VAD для ZESP: демон `asr_zesp` и модели.
Здесь лежат **исходные файлы**, а не готовый архив — архивы собирает CI на каждый
тег и публикует в [Releases](../../releases/latest).

## Состав

| Путь | Что |
|---|---|
| `bin/asr_zesp_aarch64` | демон для колонок (linux/arm64, musl static) |
| `model/model.bin` | акустическая модель (NNASR1, fp32, ~10.4 МБ) |
| `model/vocab.txt` | словарь |
| `model/vad.bin` | модель VAD (81 признак, ~310 КБ) |

Целевая платформа — **только колонки**: linux/arm64.

`asr_zesp` — один бинарь на четыре режима, ZESP использует `--tcp` (реплика
Ogg/Opus → строка текста) и `--vad` (поток Ogg/Opus → `E`/`N`).
Проект обучения и сборки — `C:\nnom\asr_zesp` (`runtime/build_runtime.ps1`,
`export_runtime.py`, `export_vad.py`).

## Релизы

На пуш тега `v*` workflow собирает архив:

```
asr_linux_arm64.tar.gz     asr_zesp, model.bin, vocab.txt, vad.bin
```

Внутри архива — плоский список файлов: ZESP ищет `asr_zesp` **и модели рядом с
ним** (`<dir exe>/../asr`). Каталог `asr` на устройстве — ровно распакованный
архив.

## Как нода проверяет свежесть

Версия — это тег релиза, отдельный sha256 не нужен. Нода хранит установленный
тег и сравнивает его с актуальным. Актуальный берётся одним запросом, без API и
без токена:

```
HEAD https://github.com/DJONvl/ZESP_asr/releases/latest
→ 302 Location: https://github.com/DJONvl/ZESP_asr/releases/tag/v0.1.1
```

Совпало — ничего не качаем. Другой тег — качаем
`releases/download/<tag>/asr_linux_arm64.tar.gz`, распаковываем, записываем тег.
Тег меняется только когда меняются файлы, поэтому «новый релиз ZESP» здесь
ничего не значит.

Сборка детерминированная (`tar --sort=name --mtime=@0 --owner=0 --group=0
--numeric-owner`, `gzip -n`): одинаковые файлы дают одинаковый архив.

## Как выпустить новую версию

```powershell
# заменить файлы в bin/ и/или model/
git add -A
git commit -m "model: <что изменилось>"
git push
git tag v0.1.1
git push origin v0.1.1
```

Workflow `asr-release` соберёт архивы, проверит их и опубликует релиз.
Прогнать без тега: Actions → `asr-release` → Run workflow, указав тег в поле
`tag` (релиз с этим тегом будет создан или обновлён).

## Примечания

- armv7 (`.2`, `.221`) пока не собран: в `C:\nnom\asr_zesp\runtime` нет
  `asr_zesp_armv7l`. Когда появится — положить в `bin/` и добавить строку в
  workflow.
- Модель обновляется редко; новый релиз нужен только когда меняются файлы.
  Новые релизы ZESP этот репозиторий не трогают.
