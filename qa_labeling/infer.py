import hydra
import numpy as np
import pytorch_lightning as pl
from data import MyDataModule
from model import QALabler
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    dm = MyDataModule(config)

    model = QALabler.load_from_checkpoint(config["inference"]["ckpt_path"])
    trainer = pl.Trainer(accelerator="cpu", devices="auto")

    rhos = trainer.predict(model, datamodule=dm)
    print(f"Test rho: {np.mean(rhos):.2f}")


if __name__ == "__main__":
    main()
