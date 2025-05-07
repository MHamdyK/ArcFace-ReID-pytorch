import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import FaceReIDDataset
from src.transforms import train_transforms
from src.model.backbone import get_pretrained_model
from src.model.net import FaceReIDModel


@click.command()
@click.option('--train-csv', required=True)
@click.option('--root-dir',   required=True)
@click.option('--epochs',     default=20, type=int)
@click.option('--batch',      default=32, type=int)
@click.option('--lr-backbone', default=5e-5, type=float)
@click.option('--lr-archead',  default=1e-3, type=float)
@click.option('--save-path',   default='models/best_model.pth')
def main(train_csv, root_dir, epochs, batch, lr_backbone, lr_archead, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = FaceReIDDataset(train_csv, root_dir, transform=train_transforms)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)

    backbone = get_pretrained_model(device)
    model = FaceReIDModel(backbone, num_classes=125).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': model.arcface.parameters(),  'lr': lr_archead,  'weight_decay': 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for imgs, labels in tqdm(dl, desc=f'Epoch {epoch}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = logits.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        scheduler.step()
        acc = correct / total
        print(f'Epoch {epoch} – loss {running_loss/total:.4f} – acc {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f'✅ Saved new best model ({best_acc:.4f})')

    print(f'Training done. Best acc = {best_acc:.4f}')


if __name__ == '__main__':
    main()
