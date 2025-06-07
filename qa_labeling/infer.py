from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from qa_labeling.pl_modules.data import MyDataModule
from qa_labeling.pl_modules.model import QALabler
from qa_labeling.utils import TARGETS


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    dm = MyDataModule(config)

    model = QALabler.load_from_checkpoint(Path(config["inference"]["ckpt_path"]))
    trainer = pl.Trainer(accelerator="cpu", devices="auto")

    preds = trainer.predict(model, datamodule=dm)

    df = pd.DataFrame(
        np.array(preds), columns=[f"target_{i}" for i in range(len(TARGETS))]
    )

    df.to_csv(Path(config["inference"]["save_path"]), index=False)


if __name__ == "__main__":
    main()
