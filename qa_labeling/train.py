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
        CustomBert(num_classes=config["model"]["num_classes"]),
        lr=config["training"]["lr"],
    )

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name="cats-and-dogs",
            run_name="conv-classifier",
            save_dir=".",
            tracking_uri="http://127.0.0.1:8080",
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
        log_every_n_steps=1,  # to resolve warnings
        accelerator="cuda",
        devices=2,
        # strategy=DDPStrategy(
        #     ddp_comm_state=powerSGD.PowerSGDState(
        #         process_group=None,
        #         matrix_approximation_rank=1,
        #         start_powerSGD_iter=5000,
        #         use_error_feedback=True
        #     ),
        #     ddp_comm_hook=powerSGD.powerSGD_hook,
        # ),
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()