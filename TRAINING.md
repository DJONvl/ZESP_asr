# Обучение модели ASR и VAD — рабочая инструкция

Этот файл — точка входа для любой новой сессии. Здесь всё: где лежит, как
обучать, как проверять, как ставить на колонку, и какие грабли уже собраны.

Проект обучения: **`C:\ZESP_asr\train`**. Готовые артефакты для колонок:
`C:\ZESP_asr\bin`, `C:\ZESP_asr\model`.

---

## 1. Что где лежит

| Путь | Что |
|---|---|
| `C:\ZESP_asr\train` | весь проект обучения (скрипты, данные, модели) |
| `C:\ZESP_asr\train\data` | датасеты, мел-кэши, метаданные |
| `C:\ZESP_asr\train\models` | чекпоинты (`model_domain4.pt` — лучший) |
| `C:\ZESP_asr\train\runtime` | C-движок (`asr.c`, `vad_model.c`), бинари, `model.bin` |
| `C:\ZESP_asr\train\runtime_domain4` | экспорт лучшей модели (NNASR1) |
| `C:\ZESP_asr\train\live` | исходные ogg с колонки (сырьё, 199 каталогов) |
| `C:\ZESP_asr\bin` | `asr_zesp_aarch64` для релиза |
| `C:\ZESP_asr\model` | `model.bin`, `vocab.txt`, `vad.bin` для релиза |

Окружение: системный **Python 3.11** (`C:\Users\Vlad\AppData\Local\Programs\Python\Python311\python.exe`),
в нём torch 2.11+cu128 и torchaudio. В `train\.venv` — тот же Python с
`--system-site-packages`; можно пользоваться им: `train\.venv\Scripts\python.exe`.
GPU: RTX 4090 Laptop 16 ГБ.

---

## 2. Быстрый старт (типовой цикл)

```powershell
cd C:\ZESP_asr\train
$py = ".venv\Scripts\python.exe"

# 1) новые записи с колонки -> wav + meta (второй чанк Ogg!)
& $py import_live_chunks.py

# 2) мел-кэш с нормализацией уровня
& $py precompute_mels.py --meta data\live_nov21.csv --normalize

# 3) дообучение от текущей лучшей модели
& $py distill_soft.py --init models\model_domain4.pt --limit 5000 `
    --extra-live data\live_nov21.csv --extra-meta data\extra_train.csv `
    --extra-limit 40000 --live-weight 40 --epochs 4 --batch 16 --lr 1e-5 `
    --lam 0 --amp --out models\model_next.pt

# 4) оценка
& $py eval_ckpt.py models\model_domain4.pt models\model_next.pt --limit 5000

# 5) экспорт
& $py export_runtime.py --ckpt models\model_next.pt --out runtime_next

# 6) деплой на колонку — см. §6
```

---

## 3. Данные

### 3.1 Источники

| Корпус | Где | Объём | Метки |
|---|---|---|---|
| Записи с колонки (новые) | `train\live\<дата>_<id>\audio-*.ogg` | 199 каталогов | `meta.json` → `yandex_text` |
| Живые записи .128 | `data\live\*.wav` + `data\live_meta.csv` | ~2000 фраз | человеческие + GigaAM |
| Golos farfield | `data\golos_farfield\wav` + `meta.csv` | 125 ч, 116k | человеческие |
| Синтез (TTS) | `data\wav` + `data\combined_meta.csv` | 25k фраз | текст фразы |

### 3.2 ⚠ Ogg с колонки = два чанка

Колонка пишет pre-roll, затем после споттера новый `OpusHead` в том же
`serial`. Проверено:

| чанк | длительность | содержимое |
|---|---|---|
| 1 | ровно 2.0 с | «алиса» (активация) |
| 2 | остальное | команда без «алиса» |

**Правильно: брать второй чанк** (`import_live_chunks.py` режет по второму
`OpusHead`). Тогда метка `yandex_text` (в ней нет «алиса») совпадает с аудио.
Если взять весь Ogg — модель учится вставлять «алиса» и CER растёт вдвое
(0.65 против 0.44).

Альтернатива — `cut_alisa.py` (резка по word timestamps GigaAM), но она грубее.

### 3.3 ⚠ Нормализация уровня обязательна

Уровни корпусов различаются на 40 дБ:

| корпус | peak | RMS |
|---|---|---|
| TTS | −3 dBFS | −23 dBFS |
| live .128 | −3 dBFS | −24 dBFS |
| **Golos farfield** | **−43 dBFS** | **−61 dBFS** |

`(m - mean)/std` **не спасает**: `clamp_min(1e-6)` применяется к power до
нормализации, и у тихого корпуса 48 % мел-бинов обрезается (у TTS — 0 %).
Лечится флагом `precompute_mels.py --normalize` (RMS → −24 dBFS, защита от
клипа). После него floor 48 % → 0 %.

### 3.4 ⚠ Метки: json обычно верен, но проверять надо

`yandex_text` в `meta.json` — это лог **облака**, а не нашей модели: на записях
23.11 он совпадает с GigaAM (264 из 428 — полное согласие), тогда как наша
локальная ASR на том же аудио ошибается чаще. То есть доверять json можно, но
не слепо.

Диагностика — `compare_labels.py` (пишет `data/labels_compare.csv` с колонками
`json|giga|ours|dur|rms|flag`). Флаги:

| flag | смысл | что делать |
|---|---|---|
| `all_agree` | все трое согласны | брать |
| `all_differ` | расходятся все | **брать**: json и GigaAM обычно близки, ошибается наша |
| `ours_matches_json` | наша и json согласны | брать |
| `asr_agree_json_diff` | обе ASR согласны, json нет | брать (json устарел) |
| `giga_junk` | GigaAM слышит шум/пусто | **выбросить** |

Правило: `keep = flag != "giga_junk"`. На 23.11 это 383 пары из 428.

⚠ Не использовать строгий консенсус (CER ≤ 0.3 по всем парам): он выбрасывает
годные короткие фразы, где одно слово меняет CER на 0.3+. Такой вариант
(`consensus_labels.py`) дал 323 вместо 383.

`filter_labels.py` (только GigaAM против json) — ещё грубее, оставлен для
сравнения.



---

## 4. Обучение

### 4.1 С нуля: `train_ctc.py`

```powershell
py train_ctc.py --epochs 30 --batch 16 --lr 3e-4 --amp
```

Архитектура: Conv1d стек (80 → 128 → 192 → 256) + BiGRU(256) + FC, **2.59M
параметров**, субсамплинг 4×, CTC, char-level (36 символов: blank, пробел,
запятая, 33 буквы). Сохранение — `--out` (по умолчанию `models/model.pt`,
лучший по CER).

Полезные флаги:

| флаг | смысл |
|---|---|
| `--meta` | свой корпус (по умолчанию `combined_meta.csv` с TTS) |
| `--out` | куда сохранить лучший чекпоинт |
| `--holdout-speaker -1` | не резать по «спикерам», а взять случайные `--val-limit` |

**Обучение только на live** (без TTS и Golos): `data/live_train_only.csv`
собирается из `live_meta.csv` (без val-части) + `live_nov23_v2_train.csv`.
На 23.11 это 1901 фраза, ~10 часов. Для 2.59M модели это мало (нужно 30+ ч),
поэтому такой вариант проверяется как эксперимент, а рабочая модель —
дообученная (`distill_soft.py`).


### 4.2 Дообучение: `distill_soft.py`

Основной инструмент. Умеет:

| флаг | смысл |
|---|---|
| `--init` | стартовый чекпоинт (всегда дообучаем от лучшего) |
| `--limit 5000` | сколько TTS-фраз брать |
| `--extra-live` | **новые живые записи: только в train**, не в val |
| `--extra-meta/--extra-limit` | Golos или другой корпус |
| `--live-weight 40` | перекос живых фраз (WeightedRandomSampler) |
| `--lam 0.5 --temp 2 --conf 0.9` | soft-KL дистилляция GigaAM (см. §7) |
| `--amp` | fp16 на GPU |

### 4.3 ⚠ Домены конфликтуют (catastrophic forgetting)

Каждый корпус тянет модель в свою сторону:

| обучено на | .128 CER | .151 CER |
|---|---|---|
| только .128 live | **0.036** | 0.65 |
| + Golos 25k без баланса | 0.186 | 0.35 |
| + Golos + live ×30 | 0.078 | — |

Правила:

1. **Всегда** `--live-weight 30…50`, если добавляете крупный корпус.
2. Смотреть CER на **двух** валидациях: своя (человеческая) и новый домен.
3. Сохраняется лучший по живому CER — не по TTS.

### 4.4 VAD

```powershell
py vad_data.py --n-train 4000 --n-val 400   # метки по энергии, whisper не нужен
py train_vad.py --epochs 30                  # AUC ~0.899
py export_vad.py                             # -> runtime/vad.bin
```

Вход VAD: 80 log-mel (per-utterance) + **log-RMS кадра** 81-м каналом, поэтому
он чувствителен к уровню входа сильнее, чем ASR.

---

## 5. Оценка

| скрипт | что считает |
|---|---|
| `eval_ckpt.py CKPT...` | CER на TTS-holdout и живом val |
| `eval_wer.py CKPT... --teacher` | CER **и WER**, плюс учитель GigaAM |
| `eval_human_live.py` | 169 человеческих фраз (чистый тест) |
| `eval_golos_test.py` | Golos test (1915 фраз, вне обучения) |
| `eval_runtime.py runtime_station10 runtime_domain4` | CER/WER C-рантайма через `asr.exe` |
| `eval_vad_live.py`, `eval_vad_missed.py` | пропуски речи VAD |

⚠ При оценке фраз с колонки помните: модели добавляют «алиса», а `yandex_text`
её не содержит. Для коротких фраз это +25 % WER. `eval_nov19_alisa.py`
показывает оба варианта.

### Текущие результаты

Замер на holdout нового домена .151 (75 фраз) и на прежнем .128 (169 фраз):

| модель | .151 CER / WER | .128 CER / WER |
|---|---|---|
| `model_domain4` | 0.3063 / 0.6164 | 0.0359 / 0.1149 |
| `model_domain6` | 0.2799 / 0.5733 | 0.0378 / 0.1297 |
| **`model_domain7`** (на .151) | **0.2799 / 0.5819** | **0.0350** / 0.1189 |

На живой колонке .151 (файловый режим `asr_zesp`, те же 75 фраз):
`domain4` — WER 0.6336, `domain7` — **WER 0.5905**.

VAD: ложных пропусков речи 0 % (было 1.8 %).

Урок по балансу: `-live-weight 100` + `lr 5e-6` + 2 эпохи дали компромисс
(новый домен лучше, старый почти без потерь), тогда как `-live-weight 40` +
`lr 1e-5` + 3 эпохи тянули модель в новый домен и портили .128.


---

## 6. Экспорт и деплой на колонку

### 6.1 Экспорт

```powershell
py export_runtime.py --ckpt models\model_domain4.pt --out runtime_domain4
py export_vad.py          # runtime/vad.bin
```

### 6.2 Колонка .151 (Station Lite, aarch64, 238 МБ RAM)

Рабочая папка ASR на устройстве — **`/opt/zesp/asr`** (это путь, содержимое
обновляем):

```
/opt/zesp/asr/asr_zesp   f8e5b843722fa6da0043ff264d986cf7
/opt/zesp/asr/model.bin  модель (сейчас domain4, 1f9d5612a40f9aadb3990573c424f2c0)
/opt/zesp/asr/vad.bin    6e14a3b1a9d573137d1efb0495ae0e07
/opt/zesp/asr/vocab.txt  ca7be23d9751dfa2f4ef23829fee5db4
```

Деплой (с dev):

```powershell
cmd /c "ssh root@192.168.1.151 ""cat > /opt/zesp/asr/model.bin""" < runtime_domain4\model.bin
ssh root@192.168.1.151 "md5sum /opt/zesp/asr/model.bin; /etc/init.d/S99zesp restart"
```

Копию кладём и в `/bkupgrade/opt/zesp/asr/model.bin` — на неё указывает
симлинк `/opt/asr`.

### 6.3 ⚠ Ловушки деплоя на .151

1. **`asrBinPath()` ищет `<родитель ZESP>/asr`** = `/opt/asr`. Реальный каталог
   — `/opt/zesp/asr`, поэтому нужен симлинк:
   `ln -sfn /bkupgrade/opt/zesp/asr /opt/asr`.
2. **`Alice_Mode` должен быть `local`** (`transit` — ASR не запускается).
3. **Кластерная политика перезаписывает `jsconfig.txt`** при старте (роль
   `leaf`). Менять надо `desktop/cluster_services.local.json`:
   ```json
   { "asr": [{"provider":"local","service":"asr_zesp","url":"tcp:127.0.0.1:9874","priority":0}],
     "vad": [{"provider":"local","service":"asr_zesp","url":"tcp:127.0.0.1:18092","priority":0}] }
   ```
4. **Останавливать только `/etc/init.d/S99zesp stop`**. Ручной запуск
   `zesp-daemon.sh` плодит копии, они дерутся за 443/18080.
5. **Один рестарт достаточен**: `applyVoiceConfig` делает hot-apply ASR
   (`internal/app/app.go`). Раньше требовалось два.
6. **VAD на 18092 понимает Ogg** (как и silero на 18193). ZESP шлёт Ogg —
   совместимо. Если VAD «не слышит», проверять: слушается ли порт 18092
   (демон мог стартовать без `--vad`).

### 6.4 Проверка после деплоя

```powershell
ssh root@192.168.1.151 "ps w | grep asr_zesp; netstat -ln | grep -e 9874 -e 18092"
```

Ожидается:

```
/opt/zesp/asr/asr_zesp --model model.bin --vocab vocab.txt --tcp 9874 --vad 18092
tcp 0 0 0.0.0.0:18092 LISTEN
tcp 0 0 0.0.0.0:9874  LISTEN
```

Затем сказать фразу и посмотреть `/tmp/asr_tcp.log` — там должны быть
`vad p=… sp=1 …`, `vad seg E speech=…`, и строка
`tcp: N B ogg -> gain 512.0 -> "текст"`.

### 6.5 Как обновлённая модель попадает в git и в релиз

В git репозитория `ZESP_asr` лежат **только готовые артефакты**, обучение
(`train/`) исключено через `.gitignore`:

| отслеживается | назначение |
|---|---|
| `model/model.bin`, `model/vocab.txt`, `model/vad.bin` | модели для колонок |
| `bin/asr_zesp_aarch64` | демон (linux/arm64) |

CI (`.github/workflows/release.yml`) на тег `v*` берёт ровно эти файлы и
собирает `dist/asr_linux_arm64.tar.gz`.

**Одной командой:**

```powershell
cd C:\ZESP_asr

# только подготовить файлы (без коммита)
.\publish.ps1 -Ckpt models\model_domain4.pt

# модель + VAD, затем коммит, тег и пуш -> CI соберёт релиз
.\publish.ps1 -Ckpt models\model_domain4.pt -Vad -Tag v0.2.0 -Message "model: domain4"
```

Что делает `publish.ps1`:

1. `export_runtime.py` → `model/model.bin` + `model/vocab.txt` (всегда);
2. `-Vad` → `export_vad.py` → `model/vad.bin`;
3. `-Bin` → `runtime/build_runtime.ps1 -Target aarch64` → `bin/asr_zesp_aarch64`
   (нужно **только** если менялся C-код: `asr.c`, `vad_model.c`, `ws_server.c`);
4. печатает md5 всех артефактов и `git status`;
5. с `-Tag` — `git add model bin`, commit, tag, push.

Без `-Tag` коммит не делается: можно проверить md5 и решить вручную.

**Важно:** если менялась только модель, бинарник не пересобирается — CI
упакует старый `asr_zesp` с новой `model.bin`, это нормально (формат NNASR1
не менялся).


---

## 7. Что уже проверено и не работает

| гипотеза | результат |
|---|---|
| soft-KL от GigaAM (λ 0.1…1.0, T 1/2, фильтр уверенности) | **не помогает**: 6 конфигураций ≤ baseline |
| причина | сжатие 90× (232M → 2.59M) + обучение на тех же фразах |
| Golos без баланса доменов | улучшает TTS, **портит** живую речь |
| обрезка «алиса» по таймстемпам | работает, но грубее, чем второй чанк |
| TTS-корпус как основная метрика | обманчив: CER падает, WER почти нет |
| **обучение с нуля только на live** (1901 фраза, ~10 ч, 30 эп) | **хуже дообучения**: .151 WER 0.621 против 0.582, .128 — 0.405 против 0.119; словарь урезался до 35 символов |


Учитель GigaAM (232.6M, 465 МБ fp16) на этих данных: CER 0.13 TTS / 0.065 live —
лучше нашей модели, но в 90 раз больше и на колонку не влезает (int8 233 МБ).

---

## 8. Чек-лист

**Перед обучением**
- [ ] новые записи импортированы **вторым чанком** (`import_live_chunks.py`);
- [ ] мел-кэши посчитаны с `--normalize`;
- [ ] в `--extra-live` переданы новые записи (они не попадают в val);
- [ ] `--live-weight` ≥ 30, если добавлен крупный корпус;
- [ ] старт от лучшего чекпоинта (`--init`).

**Перед деплоем**
- [ ] CER/WER проверены на `.151` **и** на `.128` (не только на одной);
- [ ] `eval_runtime.py` подтвердил, что C-рантайм даёт те же цифры;
- [ ] md5 экспортированной модели записан;
- [ ] на колонке сделан бэкап `model.bin` в `/bkupgrade`.

**После деплоя**
- [ ] `ps` показывает `--vad 18092`;
- [ ] порты 9874/18092 слушают;
- [ ] фраза распознаётся (лог `/tmp/asr_tcp.log`);
- [ ] условия `Alice_Mode=local` и `cluster_services.local.json` на месте.
