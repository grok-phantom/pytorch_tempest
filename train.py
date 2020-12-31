import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj, flatten_omegaconf, convert_to_jit
from src.utils.utils import set_seed, save_useful_info

warnings.filterwarnings('ignore')


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        new_dir:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    run_name = os.path.basename(os.getcwd())
    hparams = flatten_omegaconf(cfg)

    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))
    callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint.params))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg)
    dm = load_obj(cfg.datamodule.data_module_name)(hparams=hparams, cfg=cfg)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model:
        if cfg.general.save_best:
<<<<<<< HEAD
            best_path = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
            # extract file name without folder and extension
            save_name = best_path.stem
            model = model.load_from_checkpoint(best_path, hparams=hparams, cfg=cfg, strict=False)
            model_name = f'saved_models/best_{save_name}.pth'
=======
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder
            save_name = os.path.basename(os.path.normpath(best_path))
            model = model.load_from_checkpoint(best_path, hparams=hparams, cfg=cfg, strict=False)
            model_name = f'saved_models/{save_name}'.replace('.ckpt', '.pth')
>>>>>>> upstream/master
            torch.save(model.model.state_dict(), model_name)
        else:
            check_dir('saved_models')
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)

    if cfg.general.convert_to_jit:
        convert_to_jit(model, save_name, cfg)


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    check_dir('logs')
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
