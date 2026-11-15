# QualiEdge AI Inspector

**Developed by Ammad Aslam** · © 2026 Ammad Aslam. All rights reserved.

Visual quality inspection for small and mid-sized manufacturers. Point any camera at your parts, teach the
app what a good part looks like, and it tells you whether a part, or a whole batch, contains defects.
Uncertain parts go to a person instead of being guessed.

**Live app:** https://engr-dr-ammad.github.io/qualiedge-ai-inspector/

Open it on a phone, tablet or PC, allow camera access, and it can be installed to the home screen (PWA).
Training and inspection run entirely in the browser. Photos never leave the device.

## What it does

| Feature | How |
|---|---|
| Any camera | Phone (front/back), laptop webcam, USB or industrial cameras that appear as a webcam (UVC). Camera picker, flashlight toggle where supported. Photo files work too. |
| Learns from good parts only | About 10+ good photos define an inspection recipe. Each part is compared with the good parts in MobileNet feature space (k-nearest-neighbour distance, normalised by leave-one-out spread of the good parts: robust median/MAD). |
| Learns defect types | Add 3+ defect photos (e.g. *missing clip*, *scratch*) and a neural classifier is trained on top (transfer learning, class-balanced, augmented). |
| Three-way decision | **PASS / REVIEW / REJECT** with adjustable limits for deviation (σ) and defect probability. |
| Batch / tray inspection | Inspect parts one by one into a lot, or set *rows × columns* to judge a whole tray in one photo. A batch fails if any part is rejected and needs review if any part is uncertain. |
| Live mode | Continuous inspection with coloured boxes on the camera view. |
| Human in the loop | Operators confirm or overrule each part; decisions can be added to the training set (active learning). |
| Honest evaluation | Leave-one-out and grouped cross-validation: good parts passed, false rejects, defects caught, **defects missed**, and a suggested pass limit. |
| Fixed-position (jig) mode | Optional lighting-normalised pixel comparison with the good reference, for parts in a fixture. |
| Recipes | One recipe per product, each with its own photos, model and thresholds. Export/import (`.qerecipe.json`) to back up or move between devices. |
| Reports | Batch history, operator decisions, CSV and JSON export, printable report (Print → PDF). |
| Offline | Service worker caches the app and the feature network after the first load. If the network cannot be downloaded, the app falls back to classic pixel features. |

## Quick start

1. **Train tab:** start the camera, place a good part inside the orange frame, press **+ Good**. Repeat for at least 10 parts
   (use **Burst** for several shots, move the part slightly between shots). Optionally add defective parts with **+ Defect**
   and a defect type.
2. Press **Train model**, then **Evaluate** to see how it would have done on photos it never saw.
3. **Inspect tab:** enter a lot ID, **Start batch**, and press **Inspect** (or the space bar) for each part, or switch on **Live**.
4. **Batches tab:** review uncertain parts, export the report.

For trays, set rows and columns in **Settings → Inspection frame** before taking training photos; each cell becomes one sample.

## Repository layout

```
app/                       Web app (deployed to GitHub Pages; no build step)
  index.html, css/, js/    UI, camera, feature extraction, decision engine, storage
  vendor/tf.min.js         TensorFlow.js 4.22.0
  sw.js, manifest          Offline support and installable app
tests/                     Synthetic test images and an end-to-end browser test (fake camera)
qualiedge-milestone-1.zip  Native desktop app foundation (Python/PySide6), Milestone 1
.github/workflows/         Deploys app/ to the gh-pages branch on every push to main
```

## Run locally

```bash
python -m http.server 8000 --directory app
# open http://localhost:8000  (camera access works on localhost and HTTPS)
```

## Test

```bash
pip install playwright pillow && python -m playwright install chromium
python tests/make_test_images.py
python -m http.server 8765 --directory app &
python tests/e2e.py
```

## Limitations

- Accuracy depends on your parts, camera and lighting: validate with **Evaluate** and on real production parts before relying on it.
- Keep lighting and distance reasonably consistent; very fine or textured-surface defects may need a closer view or the higher-accuracy network.
- Data lives in the browser's storage on that device. Export recipes to back them up; clearing site data deletes them.
- This is an inspection aid; it does not replace qualified quality control.

## Licence

Proprietary. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
