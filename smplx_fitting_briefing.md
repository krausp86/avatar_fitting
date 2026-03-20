# SMPL-X Avatar System – Vollständiges Briefing

## Kontext und Gesamtziel

Dieses Dokument beschreibt ein vollständiges System zur Erstellung, Verwaltung
und Generierung physikalisch simulierbarer 3D-Avatare. Es besteht aus drei
Hauptkomponenten:

1. **Fitting-Pipeline** – Avatare aus MP4-Videos extrahieren
2. **Avatar-Editor** – Gefittete Avatare interaktiv bearbeiten (Web-UI)
3. **Generative Pipeline** – Neue Avatare aus Foto, Text oder Parametern erstellen

Langfristiges Ziel: Ein Body Foundation Model das alle drei Komponenten
vereint. Jeder gefittete Avatar ist ein Trainingspunkt für dieses Modell.

---

## Teil 1: Datenmodell

### Avatar-Dateistruktur

```
person_{id}/
    metadata.json               # β, β_face, Scores, Kamera, Kleidungstyp
    poses.npz                   # θ_t, ψ_t, θ_jaw_t, θ_eyes_t, T_bones_t
    geometry.npz                # ΔV_static, ΔV_face, Cage-Struktur, bone_idx
    physics.npz                 # mass_j, k_spring_j, k_damp_j
    mvc_weights.npz             # Sparse W-Matrix (10475 x N_cage)
    albedo_body.png             # 1024x1024
    roughness_body.png
    specular_body.png
    subsurface_body.png
    displacement_body.png
    albedo_face.png             # 2048x2048
    roughness_face.png
    displacement_face.png
    iris_texture.png            # 512x512
    teeth_geometry.npz
    genital_mesh_state0.npz     # Anatomisches Sub-Mesh Zustand 0
    genital_mesh_state1.npz     # Anatomisches Sub-Mesh Zustand 1
    alpha_t.npz                 # Zustandsparameter α pro Frame (männlich)
    visibility.npz              # Sichtbarkeits-Masken pro Frame
```

### metadata.json Schema

```json
{
    "person_id": "uuid",
    "β": [...],
    "β_face": [...],
    "β_diversity_score": 0.73,
    "clothing_type": "casual",
    "biological_sex": "male|female|other",
    "has_state_variation": true,
    "alpha_range_observed": [0.0, 0.8],
    "video_source": "...",
    "camera_intrinsics": {...},
    "fitting_quality": {
        "stage1_iou": 0.91,
        "stage2_residual_mean": 0.003,
        "stage4_flow_error": 0.12
    }
}
```

### Speicherbedarf

```
Geometrie (sparse MVC):    ~0.5 MB
Texturen (PNG):            ~13  MB
  Körper-Maps (5x 1024²):   3.8 MB
  Gesicht-Maps (3x 2048²):  9.0 MB
  Iris (512²):              0.2 MB
Anatomisches Sub-Mesh:     ~0.5 MB
Metadaten:                 ~10  KB

GESAMT statisch:           ~14  MB

Animations-Daten:          ~4.3 KB pro Frame
  10 Sekunden:              1.3 MB zusätzlich
  1 Minute:                 7.6 MB zusätzlich
```

---

## Teil 2: Fitting-Pipeline

### Eingabe (bereits vorhanden aus UI)
- MP4-Video (30fps, beliebige Länge)
- Segmentierungsmasken pro Frame und Person (M_t)
- Kleidungssegmentierung (SCHP)
- Personen-IDs mit Merge-Funktionalität
- Kamera-Intrinsics

### Pipeline – 7 Stages

```
Stage 1:    SMPL-X Fitting          → β, θ_t, T_bones_t
    ↕ iteriert 2-3x
Stage 2:    Statische Offsets       → ΔV_static
Stage 1.5:  Face Refinement         → β_face, ψ_t, ΔV_face  (parallel zu Stage 2)
Stage 2.5:  Textur + Appearance     → alle UV-Maps, Iris, Zähne, Sub-Mesh
Stage 3:    Signifikanz + Cage      → Cage-Topologie, MVC Weights
Stage 4:    Physikparameter         → mass_j, k_spring_j, k_damp_j
Stage 5:    PriorMLP                → β → Physikparameter (über Personen)
```

---

### Stage 1 – SMPL-X Fitting

**Ziel:** β und θ_t schätzen ohne Deformationen zu kontaminieren.

```python
static_frames = select_frames_by_joint_velocity(threshold=0.05)

loss = L_keypoint(project(joints_smplx), keypoints_2d)
     + L_silhouette(render(V_smplx), M_t)
     + L_shape_prior(β)
     + L_pose_prior(θ)

# output.A liefert T_bones_t direkt (55 x 4x4)
```

Abhängigkeiten: `smplx`, PyTorch3D, DWPose/MediaPipe, PARE/PyMAF-X.
Stage 1 und 2 iterieren 2-3x bis Konvergenz.

---

### Stage 1.5 – Face Refinement (parallel zu Stage 2)

```python
face_frames    = crop_face_region(frames, smplx_output.joints['head'])
β_face_init    = deca_or_mica_estimate(face_frames[0])

β_face, ψ_t, θ_jaw_t, θ_eyes_t, ΔV_face = face_refinement_optimizer(
    face_frames, init_β_face=β_face_init,
    renderer=pytorch3d_renderer, uv_res=2048
)

iris_texture   = bake_iris_texture(face_frames, β_face, resolution=512)
teeth_geometry = fit_teeth(face_frames[open_mouth_frames], β_face)
```

Modelle: [DECA](https://github.com/yfeng95/DECA),
[MICA](https://github.com/Zielon/MICA),
[EMOCA](https://github.com/radekd91/emoca)

---

### Stage 2 – Statische Vertex-Offsets

```python
R_t = V_observed(t) - V_smplx(β, θ_t)

for cluster c in pose_clusters:
    ΔV_static[c] = mean(R_t for t in cluster_c)

loss += laplacian_smooth(ΔV_static) + L2_norm(ΔV_static)
```

---

### Stage 2.5 – Textur + Appearance (PBR)

#### 2.5a – UV-Baking

```python
for t in visible_frames:
    pixel_to_uv = pytorch3d_rasterize(V_smplx + ΔV_static, faces, uv_coords, camera_t)
    color_t     = grid_sample(frame_t, pixel_to_uv)
    w = visibility_mask_t * dot(surface_normals_t, view_dir_t).clamp(0)
    texture_map += color_t * w
    weight_map  += w

texture_map = torch.median(frame_stack, dim=0).values  # robust gegen Beleuchtung
```

#### 2.5b – PBR-Maps schätzen

```python
roughness_map    = estimate_roughness(texture_stack, normal_map)
specular_map     = estimate_specular(texture_stack, view_dirs, light_dirs)
subsurface_map   = subsurface_prior(β, albedo_map)
displacement_map = shape_from_shading(texture_stack, normal_map)
```

#### 2.5c – Super-Resolution

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (4x/8x),
alternativ [SUPIR](https://github.com/Fanghua-Yu/SUPIR) bei sehr schlechter Qualität.

#### 2.5d – Inpainting

[Paint3D](https://github.com/OpenTexture/Paint3D) oder Body-Normal-Prior Fallback
für nicht-beobachtete Regionen.

#### 2.5e – Konsistenzoptimierung / Anchoring (kritisch)

```python
T = nn.Parameter(texture_complete.clone())
for t in visible_frames:
    rendered = pytorch3d_render_texture(V_smplx + ΔV_static, T, camera_t)
    L_photo  = MSE(rendered * visibility_mask_t, frame_t * visibility_mask_t)
    L_smooth = laplacian_smooth(T)
    L_anchor = MSE(T[~unobserved_mask], texture_complete[~unobserved_mask]) * λ_anchor
    (L_photo + L_smooth + L_anchor).backward()
    optimizer.step()
```

Ohne diesen Schritt verfälschen halluzinierte Details beobachtete Bereiche.

#### 2.5f – Anatomisches Sub-Mesh (Pflichtmodul)

Alle Körperregionen werden gleich behandelt – keine Ausschlussmaske.
Der Intimbereich wird als separates hochauflösendes Sub-Mesh modelliert:

```python
# Sub-Mesh Fitting (wie Zähne / Iris)
genital_mesh_0, genital_mesh_1 = fit_anatomical_submesh(
    frames, β, θ_t,
    region_vertices = get_vertex_mask(smplx_model, 'genital_region'),
    uv_res          = 1024
)

# Männlich: kontinuierlicher Zustandsparameter α ∈ [0, 1]
# α als optimierbarer Parameter pro Frame (nur wenn biological_sex == 'male')
α_t = nn.Parameter(torch.zeros(T_frames).clamp(0, 1))

loss_alpha = L_silhouette(render(V_state(α_t)), M_t)
           + L_smooth_alpha(α_t)   # keine abrupten Sprünge
           + L_prior_alpha(α_t)    # Ruhezustand ist a-priori wahrscheinlicher

# Geometrie-Interpolation zwischen Zuständen (ARAP für plausible Zwischenzustände)
V_state(α) = blend_shapes(genital_mesh_0, genital_mesh_1, α, method='ARAP')

# MVC-Matrizen für beide Zustände
W_0 = compute_mean_value_coordinates(V_smplx_tpose, cage_state0)
W_1 = compute_mean_value_coordinates(V_smplx_tpose, cage_state1)
W(α) = (1 - α) * W_0 + α * W_1   # interpoliert
```

Textur des Sub-Mesh wird wie Körper gebakt (2.5a–2.5e).

---

### Stage 3 – Signifikanzanalyse + Cage

```python
mean_residual = mean(norm(R_t, dim=-1), dim=t)
std_residual  = std(norm(R_t, dim=-1), dim=t)
significant   = (mean_residual > τ_mean) & (std_residual > τ_std)
cage_seeds    = KMeans(n_clusters=N_cage).fit(V_smplx[significant]).cluster_centers_

# Lokale Knochen-Frames (zwingend – nie in World Space)
for j in range(N_cage):
    bone_idx[j]  = nearest_bone(cage_seeds[j], output_tpose.joints)
    T_bone_tpose = output_tpose.A[0, bone_idx[j]]
    c_j_local[j] = (T_bone_tpose.inv() @ [*cage_seeds[j], 1.0])[:3]

# Sparse MVC: 2.4 MB → ~120 KB
W_sparse = scipy.sparse.csr_matrix(
    compute_mean_value_coordinates(V_smplx_tpose, cage_vertices_tpose)
)
```

---

### Stage 4 – Physikparameter

```python
def simulate_step(cage_vertex, T_bone, g_world, dt=1/30):
    c_rest    = (T_bone @ [*cage_vertex.c_local, 1.0])[:3]
    g_local   = T_bone[:3, :3].T @ g_world
    F_spring  = cage_vertex.k_spring * (c_rest - cage_vertex.c_current)
    F_gravity = cage_vertex.mass * g_local
    F_damping = -cage_vertex.k_damp * cage_vertex.velocity
    acc = (F_spring + F_gravity + F_damping) / cage_vertex.mass
    cage_vertex.velocity   += acc * dt
    cage_vertex.c_current  += cage_vertex.velocity * dt
    return (T_bone @ [*cage_vertex.c_current, 1.0])[:3]

# Flow Residuum als Kern-Signal
flow_rigid    = project(V_smplx(θ_t)) - project(V_smplx(θ_{t-1}))
flow_observed = RAFT(frame_t, frame_{t-1})
flow_residual = flow_observed - flow_rigid

# Chunks von 30 Frames, velocity detachen
velocity = velocity.detach()
```

---

### Stage 5 – PriorMLP

```python
class PhysicsPriorMLP(nn.Module):
    def forward(self, β, cage_local_positions, region_ids):
        x = concat(β, cage_local_positions, one_hot(region_ids))
        return log_mass, log_k_spring, log_k_damp

# Datenbedarf: ~50 Personen minimum, ~500 gut, ~5000 sehr gut
# Diversity über β kritischer als Menge
```

---

### Zwischenstand-Callbacks

```python
@dataclass
class FittingProgress:
    person_id:    str
    stage:        str            # "1", "1.5", "2", "2.5a"–"2.5f", "3", "4", "5"
    stage_name:   str
    epoch:        int
    total_epochs: int
    loss:         float
    loss_terms:   dict
    preview_jpg:  Optional[str]  # base64, alle 50 Epochen
    mesh_obj:     Optional[str]  # base64, nach Stage-Abschluss
    texture_jpg:  Optional[str]  # base64, Stage 2.5
    heatmap_jpg:  Optional[str]  # base64, Stage 3
```

| Update-Typ | Frequenz |
|---|---|
| loss + loss_terms | Jede 10. Epoche |
| preview_jpg | Jede 50. Epochen |
| mesh_obj | Stage-Abschluss |
| heatmap_jpg | Stage 3 Abschluss |

---

### Renderer: PyTorch3D

Bevorzugt gegenüber NVDiffrast: Soft Rasterization für Silhouetten-Loss,
natives Textur-Rendering, umfangreiche SMPL-X Beispiele.

---

### Personen-Tracking

- Jede Person-ID: eigene Fitting-Instanz
- Merged Personen: Masken vereinigen, Frames sortieren, als eine Person
- Mehrere Personen parallel verarbeitbar

---

## Teil 3: Avatar-Editor (Web-UI)

### Architektur

```
Browser (Three.js / Babylon.js)
    ↕ WebSocket / REST
Backend (Python / FastAPI)
    ↕
Avatar-Daten (person_{id}/)
```

Der 3D-Viewer lädt das Avatar-Mesh live und aktualisiert es bei jeder
Parameteränderung. Änderungen werden als Delta gespeichert –
die Original-Fitting-Daten bleiben unangetastet.

---

### Editierbare Komponenten

#### 1. Körperform (β)

```javascript
// Semantische Slider (nicht rohe β-Werte)
// PCA-Raum → semantische Achsen gelernt aus Datenbasis
sliders = {
    'Körpergröße':      mapped_to_β,
    'Körpergewicht':    mapped_to_β,
    'Schulterbreite':   mapped_to_β,
    'Taillenverhältnis': mapped_to_β,
    'Brustgröße':       mapped_to_β,
    // ... weitere semantische Achsen
}

// Änderung triggert:
// 1. β neu berechnen
// 2. PriorMLP: neues β → neue Physikparameter
// 3. Mesh live updaten
```

Direktes Editieren roher β-Werte als Advanced-Option (Zahleneingabe).

#### 2. Textur / Appearance

```
Farbton-Slider:      globale Hautton-Verschiebung im UV-Raum
Roughness-Slider:    globale Oberflächenrauigkeit
Markierungen:        Muttermale, Tätowierungen als Layer über UV-Map
Pinsel-Tool:         direktes Malen auf UV-Map (wie Substance Painter, simplifiziert)
Import:              eigene Textur-Map hochladen und auf UV anpassen
```

#### 3. Physikparameter

```
Pro Körperregion (Torso, Brust, Bauch, Gesäß, Arme, Beine):
    Weichheit-Slider:   k_spring (niedrig = weich, hoch = steif)
    Masse-Slider:       mass_j (beeinflusst Trägheit)
    Dämpfung-Slider:    k_damp (niedrig = schwingt lange)

Echtzeit-Vorschau:
    Bounce-Test-Button: Avatar springt kurz → Soft-Tissue-Reaktion sichtbar
    Gravitations-Test:  Avatar neigt sich → Gewebe hängt entsprechend
```

#### 4. Anatomisches Sub-Mesh

```
Geometrie-Auswahl:   Preset-Formen aus Datenbasis (anonym, statistisch)
Zustand (männlich):  α-Slider direkt einstellbar
Textur:              wie Körper-Textur-Editor
```

---

### UI-Layout (Web)

```
┌─────────────────────────────────────────────────────┐
│  [Körperform] [Textur] [Physik] [Anatomie]  Tabs    │
├──────────────────────────┬──────────────────────────┤
│                          │                          │
│   3D-Viewer              │   Parameter-Panel        │
│   (Three.js)             │                          │
│                          │   Semantische Slider      │
│   Rotation: Maus         │   oder                   │
│   Zoom: Scroll           │   Regions-spezifische    │
│                          │   Kontrollen             │
│   [Vorne][Hinten]        │                          │
│   [Links][Rechts]        │   [Reset] [Preset]       │
│                          │                          │
├──────────────────────────┴──────────────────────────┤
│  [Speichern]  [Als neu speichern]  [Export]  [Undo] │
└─────────────────────────────────────────────────────┘
```

### Technischer Stack (Editor)

```
Frontend:   Three.js (3D-Viewer) + React (UI)
            → Avatar-Mesh als GLTF/GLB geladen
            → PBR-Material direkt in Three.js
Backend:    FastAPI (Python)
            → REST-Endpoints für Parameter-Updates
            → WebSocket für Echtzeit-Physik-Vorschau
Format:     Änderungen als JSON-Delta gespeichert
            → Original-Fitting-Daten immer erhalten
```

---

## Teil 4: Generative Pipeline

### Eingabemodi (kombinierbar)

#### Modus A – Foto

```python
def generate_from_photo(photo):
    seg      = segmentation_model(photo)
    β, _     = disentanglement_encoder(photo, seg)   # clothing-invariant
    β_face   = deca_estimate(photo)
    θ        = pose_estimator(photo, β)
    texture  = appearance_encoder(photo, β)
    texture  = inpaint_occluded(texture, seg, β)
    physics  = prior_mlp(β)
    return Avatar(β, β_face, θ, texture, physics)
```

#### Modus B – Textbeschreibung

```python
def generate_from_text(description):
    # Text → semantische β-Werte via CLIP-ähnliches Mapping
    # "großer muskulöser Mann, dunkle Haut"
    # → β-Vektor + Hautton-Prior

    β        = text_to_shape(description)     # trainiertes Mapping
    texture  = text_to_appearance(description) # Diffusionsmodell konditioniert auf β
    physics  = prior_mlp(β)
    return Avatar(β, texture=texture, physics=physics)
```

#### Modus C – Parameter-Sliders

```python
def generate_from_sliders(semantic_params):
    β       = semantic_to_β(semantic_params)   # wie im Editor
    texture = appearance_prior_sample(β)       # gesamplet aus Appearance-Prior
    physics = prior_mlp(β)
    return Avatar(β, texture=texture, physics=physics)
```

#### Modus D – Kombination

Alle Modi sind kombinierbar – z.B. Foto als Ausgangspunkt,
dann Sliders für Anpassungen, dann Text für Textur-Variationen:

```python
avatar = generate_from_photo(photo)       # Basis
avatar = editor.adjust_shape(avatar, sliders)  # Form anpassen
avatar = editor.restyle_texture(avatar, "sommerbräune, mehr Muskeln")  # Textur
```

---

### Generierungs-UI

```
┌─────────────────────────────────────────────────────┐
│  Neuen Avatar erstellen                             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [📷 Foto hochladen]  [✍️ Beschreiben]  [🎚️ Sliders]│
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Foto-Upload oder Textfeld oder Sliders     │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Optionale Anpassungen:                             │
│  Geschlecht: [●Mann] [○Frau] [○Andere]             │
│  Stil: [Realistisch] [Stilisiert]                   │
│                                                     │
│  [Generieren]                                       │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  Variante 1  │  │  Variante 2  │  [Mehr...]     │
│  │  (Vorschau)  │  │  (Vorschau)  │                │
│  │  [Wählen]    │  │  [Wählen]    │                │
│  └──────────────┘  └──────────────┘                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Generierung produziert 2-4 Varianten → User wählt → weiter in Editor.

---

### Realismusgrad

```
Mit aktuellem Fit erreichbar:
    Korrekte Körperform              ✓
    Realistisches Gewebe-Verhalten   ✓
    Grundappearance                  ✓
    Game-Engine Qualität             ✓

Noch nicht ohne weiteres:
    Sub-Surface Scattering (Haut)    → Rendering-Frage, kein Fit-Problem
    Filmniveau / nicht unterscheidbar → PBR-Maps sind Approximationen
    Uncanny-Valley-frei              → Augen / Zähne sind kritisch
```

PBR-Maps (Roughness, Specular, Subsurface) aus monokularem Video sind
Approximationen – ausreichend für Game-Engine-Qualität, nicht für Film-VFX.

---

## Teil 5: Abhängigkeiten

```
# Core
smplx
torch
pytorch3d
torchvision
numpy
scipy
scikit-learn

# Fitting
mediapipe              # Keypoints
pare                   # https://github.com/mkocabas/PARE
raft-stereo            # https://github.com/princeton-vl/RAFT-Stereo

# Face
deca                   # https://github.com/yfeng95/DECA
mica                   # https://github.com/Zielon/MICA
emoca                  # https://github.com/radekd91/emoca

# Texture Enhancement
realesrgan             # https://github.com/xinntao/Real-ESRGAN
paint3d                # https://github.com/OpenTexture/Paint3D
# supir                # https://github.com/Fanghua-Yu/SUPIR

# Clothing Segmentation
schp                   # https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

# Editor Backend
fastapi
uvicorn
websockets
python-multipart

# Editor Frontend (separat, npm)
three                  # 3D-Viewer
react
```

---

## Teil 6: Implementierungshinweise

1. **Stage-Reihenfolge ist zwingend.** Stage 1 und 2 iterieren 2-3x.

2. **Lokale Knochen-Frames sind kritisch.** Cage-Vertices in `output.A`,
   nie in World Space. Fehler hier macht die gesamte Physik ungültig.

3. **MVC Weights als sparse Matrix.** Dense: 2.4 MB → Sparse CSR: ~120 KB.

4. **Backprop durch Physik.** Alle Operationen differenzierbar.
   Velocity zwischen 30-Frame-Chunks detachen.

5. **Textur-Anchoring nicht weglassen.** Stage 2.5e ist kritisch –
   ohne es verfälschen halluzinierte Details beobachtete Bereiche.

6. **α-Parameter smooth halten.** Zustandsänderungen in Stage 2.5f
   müssen temporal regularisiert werden – abrupte Sprünge
   destabilisieren das Fitting und sind physikalisch nicht plausibel.

7. **Editor-Änderungen als Delta speichern.** Original-Fitting-Daten
   niemals überschreiben. Jede Editor-Session erzeugt ein separates
   Delta-File das auf die Original-Daten aufaddiert wird.

8. **Generierung: immer mehrere Varianten.** Sampling-Varianz im
   Appearance-Prior nutzen um 2-4 Varianten zu zeigen – User-Akzeptanz
   steigt stark wenn Auswahl vorhanden.

9. **β_diversity_score pflegen.** Beim Foundation Model Training auf
   gleichmäßige β-Raumabdeckung achten. Score beim Fitting berechnen
   und in metadata.json speichern.

10. **PBR-Maps sind Approximationen.** Das kommunizieren –
    Roughness, Specular, Subsurface aus monokularem Video
    nicht exakt rekonstruierbar. Für Film-VFX manuelles Nacharbeiten
    im Editor einplanen.

---

## Teil 7: Haare

Haare sind in SMPL-X nicht enthalten und werden als separates Modul modelliert.
Sie sind Pflichtbestandteil des Systems – ohne Haare ist kein realistischer Avatar möglich.

### Warum SMPL-X keine Haare hat

SMPL-X wurde aus 3D-Körperscans gelernt. Scanner erfassen Haare nicht zuverlässig
(zu hohe geometrische Varianz, kein sinnvoller PCA-Raum möglich).

### Drei Darstellungsebenen (kombinierbar)

```
Ebene 1: Mesh-basiert      → schnell, animierbar, Game-Engine-tauglich
Ebene 2: NeuralHaircut     → realistisch, aus Video rekonstruierbar
Ebene 3: Gaussian Splatting → volumetrisch, höchste Qualität, schwer animierbar
```

### Dateistruktur Haare

```
person_{id}/
    hair_mesh.npz           # Haar-Geometrie (Mesh-basiert, ~50k Vertices)
    hair_strands.npz        # Strand-Daten (NeuralHaircut Output)
    hair_albedo.png         # 512x512 Haarfarbe
    hair_alpha.png          # Opacity Map
    hair_flow.png           # Richtungs-Map (Kamm-Richtung per UV)
    hair_physics.npz        # k_spring, mass, damping pro Haar-Region
```

### metadata.json Erweiterung

```json
{
    "hair": {
        "style":              "long_straight",
        "color_rgb":          [60, 35, 15],
        "length_cm":          35,
        "recon_method":       "neuralhaircut",
        "has_reconstruction": true
    }
}
```

### Stage 2.5g – Haar-Fitting (nach Körper-Textur)

```python
def fit_hair(frames, β, θ_t, camera_t, method='mesh'):

    # Haar-Region aus Frames segmentieren
    hair_mask_t = segment_hair(frames)   # z.B. BiSeNet oder ModNet

    if method == 'neuralhaircut':
        # https://github.com/SamsungLABS/NeuralHaircut
        hair_strands = neuralhaircut_reconstruct(
            frames, hair_mask_t, camera_t
        )
        hair_mesh = strands_to_mesh(hair_strands)

    elif method == 'mesh':
        # Mesh-basiert: Preset + Deformation
        hair_mesh = fit_hair_mesh_to_silhouette(
            frames, hair_mask_t, camera_t, β
        )

    elif method == 'gaussian':
        # 3D Gaussians für Haare
        hair_gaussians = fit_hair_gaussians(
            frames, hair_mask_t, camera_t
        )

    # Textur backen
    hair_albedo = bake_hair_texture(frames, hair_mesh, camera_t)
    hair_alpha  = compute_opacity_map(hair_mesh, frames)
    hair_flow   = compute_flow_map(hair_strands or hair_mesh)

    return hair_mesh, hair_albedo, hair_alpha, hair_flow
```

### Haar-Physik (Cage-Erweiterung)

Haare haben eigene Physikparameter – deutlich weicher als Körper-Soft-Tissue:

```python
# Typische Werte
hair_physics = {
    'k_spring':  0.05,    # sehr niedrig – Haare sind sehr weich
    'mass':      0.01,    # pro Strand sehr leicht
    'k_damp':    0.3,     # mittlere Dämpfung
    'gravity_influence': 0.9,  # Haare folgen stark der Schwerkraft
}

# Regionen: Oberkopf, Seiten, Hinterkopf, Pony
# Jede Region hat eigene Parameter
```

### Haar-Prior im Foundation Model

```python
class HairPriorMLP(nn.Module):
    def forward(self, β, hair_style_embedding):
        # β: Körperform (beeinflusst Haardichte / Volumen)
        # hair_style_embedding: gelernte Style-Repräsentation
        x = concat(β, hair_style_embedding)
        return hair_mesh_params, hair_physics_params, hair_texture_prior

# Training: gefittete Haar-Daten als Supervision
# Inferenz: β + erkannter Style → vollständige Haare
```

### Empfohlene Modelle

```
Segmentierung:   BiSeNet / ModNet  – Haar-Maske aus Frame
Rekonstruktion:  NeuralHaircut     – https://github.com/SamsungLABS/NeuralHaircut
Textur:          Real-ESRGAN       – wie Körpertextur
Physik-Sim:      Position-Based Dynamics (PBD) – Standard in Game Engines
```

### Abhängigkeiten (Haare)

```
bisenet            # Hair segmentation
neuralhaircut      # https://github.com/SamsungLABS/NeuralHaircut
# optional:
gaussian-hair      # Gaussian Splatting für Haare
```

