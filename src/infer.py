import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import FaceReIDDataset, FaceTestDataset
from src.transforms import train_transforms, vgg_transform
from src.model.backbone import get_pretrained_model
from src.model.net import FaceReIDModel


def cosine_similarity(emb, gallery_tensor):
    emb = emb.unsqueeze(0)
    return F.cosine_similarity(emb, gallery_tensor)


@click.group()
def main():
    pass


@main.command('build-gallery')
@click.option('--train-csv', required=True)
@click.option('--root-dir',  required=True)
@click.option('--out',       default='models/gallery.npy')
@click.option('--batch',     default=32)
def build_gallery(train_csv, root_dir, out, batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FaceReIDDataset(train_csv, root_dir, transform=train_transforms)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)

    backbone = get_pretrained_model(device)
    model = FaceReIDModel(backbone, num_classes=125).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    gallery = {}
    with torch.inference_mode():
        for imgs, labels in tqdm(dl, desc="Gallery"):
            imgs, labels = imgs.to(device), labels.to(device)
            emb = model(imgs)
            for e, lbl in zip(emb, labels):
                lbl = int(lbl.item())
                gallery.setdefault(lbl, []).append(e.cpu().numpy())

    for lbl in gallery:
        gallery[lbl] = np.mean(np.stack(gallery[lbl]), axis=0)

    np.save(out, {'labels': list(gallery.keys()),
                  'embeddings': np.stack(list(gallery.values()))})
    print(f'✅ Gallery saved → {out}')


@main.command('predict')
@click.option('--eval-csv', required=True)
@click.option('--root-dir', required=True)
@click.option('--gallery',  required=True)
@click.option('--threshold', default=0.80, type=float)
@click.option('--out', default='submission.csv')
@click.option('--batch', default=32)
def predict(eval_csv, root_dir, gallery, threshold, out, batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FaceTestDataset(eval_csv, root_dir, transform=vgg_transform)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)

    g = np.load(gallery, allow_pickle=True).item()
    gallery_labels = g['labels']
    gallery_tensor = torch.tensor(g['embeddings'], device=device)

    backbone = get_pretrained_model(device)
    model = FaceReIDModel(backbone, num_classes=125).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    results = []
    with torch.inference_mode():
        for imgs, paths in tqdm(dl, desc="Predict"):
            imgs = imgs.to(device)
            emb = F.normalize(model(imgs))
            for e, p in zip(emb, paths):
                sim = cosine_similarity(e, gallery_tensor)
                idx = int(torch.argmax(sim))
                if sim[idx] < threshold:
                    pred = "doesn't_exist"
                else:
                    pred = f'person_{gallery_labels[idx]}'
                results.append({'image_path': p, 'pred': pred})

    # create submission
    import pandas as pd
    sub = pd.DataFrame([
        {'ID': i, 'frame': -1,
         'objects': str({'gt': r['pred'], 'image': 'test_set/' + r['image_path']}), 
         'objective': 'face_reid'}
        for i, r in enumerate(results)
    ])
    sub.to_csv(out, index=False)
    print(f'✅ submission saved → {out}')


if __name__ == '__main__':
    main()
