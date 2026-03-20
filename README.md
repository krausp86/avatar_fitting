# SOMA Avatar System

## Setup

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Umgebungsvariablen

```
AVATAR_DATA_ROOT   # Pfad zum Avatar-Datenspeicher  (default: ./avatar_data)
VIDEO_SCAN_ROOT    # Pfad zum Video-Trainingsordner (default: ./video_data)
```

## Struktur

- Videos scannen: /videos/ → "Ordner scannen"
- Personen mergen: /persons/ → Tracks auswählen → Zusammenführen
- Avatar erstellen: /avatars/create/
- Fitting starten: Avatar-Detail → "Fitting starten"
- Editor: Avatar-Detail → "Editor"

## Docker

### Schnellstart (CPU, Entwicklung)

```bash
cp .env.example .env
# VIDEO_FOLDER in .env anpassen

docker compose up --build
```

→ http://localhost:8000

### Mit GPU (CUDA)

```bash
# docker-compose.override.yml entfernen oder umbenennen
docker compose -f docker-compose.yml up --build
```

### Build-Layer Strategie

```
Layer 1  requirements-ml.txt     PyTorch, numpy, scipy       ~15 min  (einmalig)
Layer 2  requirements-fitting.txt SMPL-X, pytorch3d, tools   ~10 min  (selten)
Layer 3  requirements.txt         Django, Channels            ~1 min   (bei dep-Änderung)
Layer 4  app code                 dein Code                   ~5 sek   (bei jedem Build)
```

Solange `requirements-ml.txt` und `requirements-fitting.txt` unverändert bleiben,
dauert ein Rebuild nach Code-Änderungen nur wenige Sekunden.

### Video-Ordner mounten

```bash
# .env
VIDEO_FOLDER=/mnt/nas/training_videos
```

Der Ordner wird read-only in den Container gemountet.
Avatar-Daten werden in einem persistenten Docker-Volume gespeichert.
