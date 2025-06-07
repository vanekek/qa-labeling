from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dvc.api import DVCFileSystem
from transformers import BertTokenizer

from qa_labeling.utils import TARGETS, compute_input_arays, compute_output_arrays


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]
        if self.labels is not None:
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Old setupping
        # current_file = Path(__file__).resolve()
        # data_dir = current_file.parent.parent.parent / "data_raw"

        # train = pd.read_csv(data_dir / "train.csv")
        # val = pd.read_csv(data_dir / "val.csv")
        # test = pd.read_csv(data_dir / "test.csv")
        fs = DVCFileSystem()
        with fs.open("/data_raw/train.csv") as f:
            train = pd.read_csv(f)

        with fs.open("/data_raw/val.csv") as f:
            val = pd.read_csv(f)

        with fs.open("/data_raw/test.csv") as f:
            test = pd.read_csv(f)

        input_categories = list(train.columns[[1, 2, 5]])

        inputs_train = compute_input_arays(
            train, input_categories, self.tokenizer, max_sequence_length=290
        )
        outputs_train = compute_output_arrays(train, columns=TARGETS)
        outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
        lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
        lengths_train[lengths_train == 0] = inputs_train[0].shape[1]
        self.train_dataset = QuestDataset(
            inputs=inputs_train, lengths=lengths_train, labels=outputs_train
        )

        inputs_valid = compute_input_arays(
            val, input_categories, self.tokenizer, max_sequence_length=290
        )
        outputs_valid = compute_output_arrays(val, columns=TARGETS)
        outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
        lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
        lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
        self.val_dataset = QuestDataset(
            inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid
        )

        test_inputs = compute_input_arays(
            test,
            input_categories,
            self.tokenizer,
            max_sequence_length=512,
            t_max_len=30,
            q_max_len=239,
            a_max_len=239,
        )
        lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
        lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
        self.test_dataset = QuestDataset(
            inputs=test_inputs, lengths=lengths_test, labels=None
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
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )

    def teardown(self, stage: str) -> None:
        pass
