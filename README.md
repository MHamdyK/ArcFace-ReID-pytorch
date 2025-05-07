# ArcFace‑ReID  
Face re‑identification for surveillance using **Inception‑ResNet V1 + ArcFace**

---

<div align="center">
  <img src="docs/demo/demo.gif" width="600"/>
</div>

| metric | score | notes |
|--------|-------|-------|
| **train accuracy** | **99 %** | 125 identities, 20 epochs |
| **public‑LB accuracy** | **92 %** | completely unseen faces |

This repo converts a Kaggle notebook into a clean, reproducible pipeline:

* **Backbone:** Inception‑ResNet V1 pre‑trained on VGGFace2  
* **Head:** ArcFace (Additive‑Angular‑Margin Softmax) → 512‑D embeddings  
* **Gallery:** mean‑pooled embedding per identity  
* **Inference:** cosine similarity ≥ 0 .80, else `doesn't_exist`

---

## Project layout
~~~text
arcface-reid/
├── data/                # (git‑ignored) csv + images
├── models/
│   ├── best_model.pth
│   └── gallery.npy
├── src/
│   ├── cli.py
│   ├── datasets.py
│   ├── transforms.py
│   ├── model/
│   │   ├── backbone.py
│   │   ├── arc_margin.py
│   │   └── net.py
│   ├── train.py
│   └── infer.py
└── README.md
~~~

---

## Quick start
~~~bash
# install
pip install -r requirements.txt
pip install -e .

# train (≈ 3 h on one P100 GPU)
byteface train \
  --train-csv data/trainset.csv \
  --root-dir  data/face_identification \
  --epochs    20

# build gallery
byteface build-gallery \
  --train-csv data/trainset.csv \
  --root-dir  data/face_identification \
  --out models/gallery.npy

# predict & create submission
byteface predict \
  --eval-csv  data/eval_set.csv \
  --root-dir  data/face_identification/test \
  --gallery   models/gallery.npy \
  --out       submission.csv
~~~

---

## Training details
| hyper‑param | value | reason |
|-------------|-------|--------|
| **epochs** | 20 | converged; plateau after 17 |
| **batch** | 32 | fits 16 GB VRAM |
| **LR backbone** | 5 e‑5 | gentle fine‑tune |
| **LR ArcFace** | 1 e‑3 | train head from scratch |
| **scheduler** | CosineAnnealingLR (T_max = 20) | smooth decay |
| **augments** | crop 0.8‑1.0 · flip 0.5 · rot ±15° · colour‑jitter 0.2 | real‑world variety |

---

## Model architecture
~~~text
               +-----------------------+
Input 3×160×160|  Inception‑ResNet V1  | 512‑D embedding
-------------->+-----------------------+------+
                                           |
                                  L2‑normalise
                                           |
                                  +--------------------+
                                  |  ArcMarginProduct  |
                                  |  (s = 30, m = 0.5) |
                                  +---------+----------+
                                            |
                                     softmax logits
~~~

---

## Inference pipeline
1. **Gallery build** – average all embeddings for each ID  
2. **Cosine similarity** between query embedding and gallery matrix  
3. If `max_sim < 0.80` → assign label `doesn't_exist`

---

## References
* Deng et al., **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**  
* **facenet‑pytorch** library – Inception‑ResNet V1 + VGGFace2 weights  
* **VGGFace2** dataset – large‑scale face recognition corpus
