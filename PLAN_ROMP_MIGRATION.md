# Plan: PyTorch3D raus, ROMP rein

**Ziel:** ROMP als Initialisierer für Stage 1 einbinden (gibt β + θ als Startpunkt).
PyTorch3D komplett entfernen — war nur für Silhouette-Loss, der auf CPU sowieso deaktiviert war.

---

## Schritt 1 — ROMP-Weights herunterladen

Weights manuell herunterladen von:
> https://github.com/Arthur151/ROMP/releases

Dateiname: `ROMP.pkl` (~200 MB, ResNet-50 Backbone)

Ablegen unter:
```
models/romp/
    ROMP.pkl
```

---

## Schritt 2 — requirements-fitting.txt

```diff
+romp>=1.1.0
```

Kein weiterer Build-Aufwand — `pip install romp` reicht.

---

## Schritt 3 — docker-compose.yml

```yaml
environment:
  ROMP_MODEL_PATH: /data/models/romp/ROMP.pkl
# volumes sind bereits korrekt: ${MODELS_DIR:-./models}:/data/models:ro
```

---

## Schritt 4 — Dockerfile (GPU-Variante)

PyTorch3D-Buildzeilen entfernen:

```diff
-# pytorch3d (build from source for CUDA compatibility)
-RUN pip install \
-    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
-    || echo "pytorch3d build failed – falling back to CPU render"
```

---

## Schritt 5 — `core/fitting/romp_init.py` (neue Datei, ~60 Zeilen)

```python
def romp_init_frames(frames, model_path) -> dict | None:
    """
    Läuft ROMP über N Frames.
    Gibt zurück:
      {
        'beta':   np.ndarray (10,)    — Median über alle Frames
        'thetas': np.ndarray (N, 72)  — per-Frame [global_orient(3) + body_pose(69)]
      }
    Gibt None zurück wenn ROMP nicht verfügbar (graceful fallback).
    """
```

- Frames als Liste von numpy-Arrays (H×W×3, uint8)
- ROMP-Output: `smpl_thetas` (72-dim), `betas` (10-dim)
- Median über alle Frames für `beta` (Shape ist zeitinvariant)
- Fallback auf `None` wenn Import fehlschlägt oder model_path fehlt

---

## Schritt 6 — `core/fitting/stage1.py` anpassen

### Was raus:
- `_pytorch3d_available()` (Zeile ~1213)
- `_build_silhouette_renderer()` (Zeile ~1221)
- `_silhouette_loss_batch()` (Zeile ~1265)
- `sil_ok`-Flag + alles was daran hängt (Zeile ~185)
- `w_silhouette` aus `Stage1Config` (Zeile ~108)
- `load_masks=sil_ok` in `_load_person_frames()`-Aufruf
- Masken-Ladelogik `_load_mask_lookup` (wenn nur für Silhouette gebraucht)

### Was rein (in `run_stage1()`, nach Frame-Loading):
```python
from .romp_init import romp_init_frames
_romp = romp_init_frames(frames, settings.ROMP_MODEL_PATH)
if _romp is not None:
    _beta_init = torch.tensor([_romp['beta']], ...)
    # thetas für z_ref-Initialisierung nutzen (Median-Frame)
```

- Priorität: PersonShape > ROMP > Nullvektor
- ROMP-Thetas für `z_ref` initialisieren: Median-Frame durch VPoser enkodieren
  wenn VPoser verfügbar, sonst direkt als body_pose_ref

---

## Schritt 7 — `core/fitting/single_frame_fit.py` bereinigen

Gleiche Bereinigung wie Stage 1:
- `_pytorch3d_available()` raus (Zeile ~84)
- GPU-Silhouette-Pfad raus (Zeilen ~566–740)
- Nur CPU-Pfad oder komplett raus

---

## Schritt 8 — `avatar_system/settings.py`

```python
ROMP_MODEL_PATH = os.environ.get('ROMP_MODEL_PATH', './models/romp/ROMP.pkl')
```

---

## Reihenfolge beim Umsetzen

1. `ROMP.pkl` herunterladen → `models/romp/`
2. `pip install romp` im Container testen
3. `romp_init.py` schreiben + mit einem Testframe validieren
4. Stage 1 bereinigen (pytorch3d raus)
5. ROMP-Initialisierung in Stage 1 einbauen
6. `single_frame_fit.py` bereinigen
7. Dockerfile bereinigen
8. End-to-End testen

---

## Erwartetes Ergebnis

Stage 1 startet nicht mehr von β=0, θ=0 sondern von einem plausiblen ROMP-Schätzwert.
Die Optimierung ist dann echtes Refinement statt Suche im blinden Raum.
Auf CPU: ROMP-Inference ~0.5s/Frame, für 50 Frames ~25s — akzeptabel.
