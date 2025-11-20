import sys
sys.path.append('..')
import warnings
import optuna

from darts.models import TiDEModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from utils.data import CSVLoader, make_time_series_dict


# NOTE (2023-19-02): Optuna’s PyTorchLightningPruningCallback raises an error
# with pytorch-lightning>=1.8. Until this fixed, a workaround is proposed here.
# https://github.com/optuna/optuna-examples/issues/166#issuecomment-1403112861
class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def objective(trial):
    model = TiDEModel(
        input_chunk_length=7 * 24,
        output_chunk_length=24,
        batch_size=256,
        hidden_size=trial.suggest_categorical('hidden_size', [64, 128, 256]),
        num_encoder_layers=trial.suggest_int('num_encoder_layers', 1, 3),
        num_decoder_layers=trial.suggest_int('num_decoder_layers', 1, 3),
        decoder_output_dim=trial.suggest_categorical('decoder_output_dim', [4, 8, 16, 32]),
        temporal_decoder_hidden=trial.suggest_categorical('temporal_decoder_hidden', [32, 64, 128]),
        dropout=trial.suggest_float('dropout', 0.0, 0.5),
        use_layer_norm=trial.suggest_categorical('use_layer_norm', [True, False]),
        use_reversible_instance_norm=trial.suggest_categorical('use_reversible_instance_norm', [True, False]),
        optimizer_kwargs={
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        },
        pl_trainer_kwargs={
            'log_every_n_steps': 500,
            'max_epochs': 100,
            'accelerator': 'gpu',
            'devices': [device],
            'callbacks': [
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode='min',
                ),
                PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            ]
        },
    )

    model.fit(
        series=series,
        future_covariates=future_covariates,
        val_series=val_series,
        val_future_covariates=val_future_covariates,
        verbose=True,
        epochs=100,
        num_loader_workers=4,
    )

    loss = model.trainer.early_stopping_callback.best_score.cpu().numpy().item()

    return loss


if __name__ == "__main__":
    device = int(sys.argv[1])
    print('Device:', device)

    # Create datasets.
    csv_loader = CSVLoader('../datasets/Cambridge-Estates-Building-Energy-Archive')

    series = []
    future_covariates = []
    val_series = []
    val_future_covariates = []

    for bid in csv_loader.building_ids:
        data = make_time_series_dict(
            bid=bid,
            csv_loader=csv_loader,
            train_range=('2009-01-01', '2009-10-01'),
            val_range=('2009-10-01', '2010-01-01'),
        )

        series.append(data['train_series'])
        future_covariates.append(data['train_future_covariates'])
        val_series.append(data['val_series'])
        val_future_covariates.append(data['val_future_covariates'])

    # Create study if not exists.
    study_name = "tide-hypertune"
    study = optuna.create_study(
        study_name=study_name, storage="sqlite:///../output/assets/tide-hypertune.db",
        load_if_exists=True,
    )
                
    study.optimize(objective, n_trials=200)