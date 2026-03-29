# SOTA-Analyse: Warum das aktuelle Fitting-System kämpft

## Das Kernproblem

Das System macht folgendes:

> 12 COCO-2D-Keypoints → projiziere SMPL-X auf 2D → minimiere Abstand

Das ist ein **hochgradig nicht-konvexes, unterdeterminiertes Problem**. Für dasselbe 2D-Keypoint-Bild existieren hunderte valide 3D-Posen. Reine Optimierung from-scratch hat gegen diesen Suchraum keine Chance — das ist kein Bug, sondern ein strukturelles Problem.

---

## Was SOTA anders macht

### 1. Regression statt Optimierung

| Was wir machen | Was SOTA macht |
|---|---|
| β=0, θ=0, optimiert per-frame über hunderte Epochen | Regrediert β und θ direkt aus Bild-Features (1 Forward Pass) |

**HMR2.0, CLIFF, OSX, SMPLer-X** sind Regressionsnetze — trainiert auf Millionen Samples mit Pseudo-GT-Annotierungen. Ein einziger Forward Pass gibt direkt eine plausible Schätzung. Optimization-from-scratch als einziger Schritt funktioniert praktisch nicht zuverlässig, weil der Optimierer nie weiß "in welche Richtung" er suchen soll.

**Selbst Methoden die optimieren** (EFT, SPIN, NeuralAnnot) starten **nicht von Null** — sie nutzen ein Regressionsnetz als Initialisierer und verfeinern dann. Unser Phase-1-Warmup (50 Epochen orient+transl) arbeitet trotzdem gegen eine flache Nullpose.

---

### 2. Tiefen-Ambiguität bekommt kein Signal

Aus einer Monokamera mit unbekannter Brennweite sind Tiefe und Körpergröße vollständig ambig.

```
fx = max(W, H) * 1.2  →  Schätzung, kein Signal
```

- **CLIFF** konditioniert auf die vollständige Bildgröße + Person-Position im Bild um die Kamera zu schätzen
- **WHAM** integriert "virtuelle IMU" aus optischem Fluss → Tiefe aus Bewegung
- **SLAHMR** löst Kamera + Person gemeinsam auf Szenenebene

---

### 3. Sparse Keypoints vs. dichte Korrespondenzen

12 Joints = ~24 Zahlen pro Frame als einziges Signal. Das reicht nicht.

SOTA nutzt:

| Methode | Was sie nutzen |
|---|---|
| DensePose-basiert | UV-Maps: hunderte Punkte mit Körperoberflächenkorrespondenz |
| Silhouette konsequent | Maske schränkt Skala + globale Form massiv ein |
| Part-based Features | Regionale Bildfeatures pro Körperteil |
| Temporal Features | LSTM / Transformer über mehrere Frames |

Unser Silhouette-Loss ist **optional** und auf CPU **deaktiviert**. Das ist einer der wichtigsten Terms für Shape — er fehlt fast immer.

---

### 4. Kein Pose-Prior der wirklich zieht

VPoser ist gut, aber das Gewicht `w_pose_prior = 0.001` ist extrem niedrig gegenüber `w_keypoint = 1.0`. Bei unscharfen oder okklidierten Keypoints zieht der Prior nicht genug — der Optimizer landet in anatomisch unmöglichen Posen weil die 2D-Projektion noch "passt".

---

### 5. Kein gelerntes Kamera-Modell

SMPL-Methoden wie CLIFF und BEV schätzen die Kamera als **weak-perspective** Projektion direkt mit aus dem Bild — das vermeidet das Brennweiten-Problem komplett. Wir verwenden pinhole mit fixer Schätzung.

---

## Was konkret im nächsten System besser wird

### Stufe 1: Initialisierer einbinden (größter Einzelhebel)

Einen SMPL/SMPL-X Regressor als ersten Schritt einbinden, der direkt β und θ aus dem Bild-Crop gibt. Unsere Optimierung läuft dann nur noch als **Refinement**, nicht als Suche von Null.

Kandidaten (absteigend nach Aufwand):

| Modell | Format | Stärke |
|---|---|---|
| **4D-Humans (HMR2.0)** | SMPL, PyPI installierbar | Beste per-Frame-Qualität, temporale Konsistenz |
| **OSX** | SMPL-X, GitHub | Whole-Body inkl. Hände/Gesicht |
| **SMPLer-X** | SMPL-X, GitHub | Large-scale, bester SMPL-X Regressor aktuell |
| **CLIFF** | SMPL, GitHub | Gut bei Kameradistanz-Schätzung |

Konkret: `4D-Humans` per pip installieren, per-Frame laufen lassen, β/θ als Startwert in Stage 1 einspeisen. Das allein wird den Output dramatisch verbessern.

---

### Stufe 2: Silhouette-Loss immer aktiv

PyTorch3D auf CPU ist langsam, aber möglich. Der Silhouette-Term ist **unverzichtbar** für Body-Shape — ohne ihn können β-Parameter kaum konvergieren, weil 12 Joints die Körperform nicht einschränken.

Alternative: **Soft-Rasterizer** oder **DIB-Renderer** als leichtere PyTorch3D-Alternative.

---

### Stufe 3: Kamera-Optimierung

`focal_scale` als optimierbaren Parameter statt fixer Schätzung. Besser noch: weak-perspective Kameramodell (scale s, tx, ty) wie HMR — das eliminiert das Brennweiten-Problem komplett und ist leichter zu optimieren.

---

### Stufe 4: Temporal Smoothness als echten Prior

Derzeit: `w_temporal = 0.1` auf Adjacent-Frame-Differenz. SOTA:

- **WHAM**: Integration von Velocities aus optischem Fluss → physikalisch plausible Trajektorie
- **VIBE/TCMR**: GRU/Transformer über Framefenster, lernt Bewegungsdynamik
- Minimum: SMPL pose-space velocity prior (nicht nur raw theta-diff)

---

### Zusammenfassung: Nächstes System

```
Video
  ↓
Person-Detection + Tracking  (wie jetzt)
  ↓
ViTPose Keypoints             (wie jetzt, läuft schon)
  ↓
SMPLer-X / 4D-Humans          ← NEU: Regressor als Initialisierer
  ↓
Stage 1 Refinement            (Optimization, aber von gutem Startpunkt)
  - Silhouette-Loss immer aktiv
  - weak-perspective Kamera
  - w_pose_prior erhöhen auf ~0.01
  ↓
Stage 2+                      (wie geplant)
```

Der entscheidende Schritt ist der Regressor. Alles andere ist Feintuning.

---

## Referenzen

- [HMR2.0 / 4D-Humans](https://shubham-goel.github.io/4dhumans/) — Goel et al., CVPR 2023
- [SMPLer-X](https://caizhongang.github.io/projects/SMPLer-X/) — Cai et al., NeurIPS 2023
- [OSX](https://github.com/IDEA-Research/OSX) — Lin et al., CVPR 2023
- [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) — Li et al., ECCV 2022
- [WHAM](https://wham.is.tue.mpg.de/) — Shin et al., CVPR 2024
- [SLAHMR](https://slahmr.github.io/) — Ye et al., CVPR 2023
