from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from ..model.unet import build_unet

class Trainer:

    def __init__(self, config: dict[str, Any], device: str | None=None) -> None:
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = build_unet(config).to(self.device)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def train(self, loader: DataLoader, epochs: int=1, checkpoint_dir: str | Path | None=None) -> None:
        self.model.train()
        out_dir = Path(checkpoint_dir or self.config.get('paths', {}).get('checkpoints', 'data/checkpoints'))
        out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            running = 0.0
            steps = 0
            for batch in loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                running += float(loss.item())
                steps += 1
            path = out_dir / f'cardiac_unet_epoch{epoch + 1}.pt'
            state = self.model.state_dict()
            torch.save(state, path)
            torch.save(state, out_dir / 'cardiac_unet.pt')
            mean_loss = running / max(steps, 1)
            print(f'epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}  saved={path}')