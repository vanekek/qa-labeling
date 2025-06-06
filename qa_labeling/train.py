import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from qa_labeling.pl_modules.classifiers import CustomBert
from qa_labeling.pl_modules.data import MyDataModule
from qa_labeling.pl_modules.model import QALabler

# from pytorch_lightning.strategies import DDPStrategy
# from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)
    dm = MyDataModule(config)
    model = QALabler(
        CustomBert(config=config),
        lr=config["training"]["lr"],
        loss_weights=config["training"]["loss_weights"],
        freeze=config["model"]["freeze"],
    )

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name="qa_labeling",
            run_name="simple_bert",
            save_dir=".",
            tracking_uri=config["logging"]["tracking_uri"],
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=5,  # to resolve warnings
        accelerator="auto",
        # devices=1,
        logger=loggers,
        callbacks=callbacks,
        limit_train_batches=0.05,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
