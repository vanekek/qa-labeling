from typing import Any, Optional

import pytorch_lightning as pl
import torch

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels = None):
        
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        input_ids       = self.inputs[0][idx]
        input_masks     = self.inputs[1][idx]
        input_segments  = self.inputs[2][idx]
        lengths         = self.lengths[idx]
        if self.labels is not None:
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])
    

class MyDataModule(pl.LightningDataModule):
    def __init__(self, config, inputs, lengths, labels=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def prepare_data(self):
        pass

        # поменять на пути
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = QuestDataset(
            self.inputs, self.length, self.labels, 
        )
        self.val_dataset = QuestDataset(
            self.inputs, self.length, self.labels, 
        )
        self.test_dataset = QuestDataset(
            self.inputs, self.length, self.labels, 
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )
    
    def teardown(self, stage: str) -> None:
        pass