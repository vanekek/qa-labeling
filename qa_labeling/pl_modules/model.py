from typing import Any

import pytorch_lightning as pl
import torch



class QALabler(pl.LightningModule):
    """Module for training and evaluation models
    for the classification task
    """

    def __init__(self, model, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        data, logits = batch
        preds = self(data)

        # For nice image visualization with wandb! :)
        # sample_imgs = data[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)

        # self.logger.experiment.log(
        #     {"example_images": [wandb.Image(grid, caption="Example Images")]}
        # )

        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data, logits = batch
        preds = self(data)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        return acc.item()
    
    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)