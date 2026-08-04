"""Callback that persists the recorded event log while training runs."""

from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec


class EventLogCallback(Callback):
    """Writes the exporter's event log to `recordings/` at each epoch end.

    Long training runs are routinely stopped before `trainer.fit()` returns —
    monitored, judged, and killed. Writing only at the end means those runs
    produce no recording at all, which is how five 25v25 runs finished with
    `--record-events` set and nothing on disk.

    `EventLog.record_reset` replaces its event list, so the log always holds a
    single episode and rewriting the same path each epoch stays cheap.
    """

    def __init__(self, run_name: str, exporter: EventLogExporter) -> None:
        self.run_name = run_name
        self.exporter = exporter
        self._codec = JsonMatchCodec()

    @property
    def output_path(self) -> Path:
        """Destination file for the serialised event log."""
        return Path("recordings") / f"{self.run_name}_events.jsonl"

    def write(self) -> bool:
        """Serialise the current log. Returns False when there is nothing useful.

        A log holding only its reset event is skipped: `record_reset` replaces
        the event list, so that state means an episode has just begun and no
        steps have been recorded. Writing it would replace a usable recording
        with a single frame.
        """
        if len(self.exporter.log) <= 1:
            return False
        self.output_path.parent.mkdir(exist_ok=True)
        self.output_path.write_bytes(self._codec.encode(self.exporter.log))
        return True

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Persist the most recent episode so a killed run still leaves a log.

        Deliberately hooked to epoch *start* rather than epoch end. Lightning
        invokes callback `on_train_epoch_end` before the LightningModule's, and
        this project runs its evaluation episodes in the module's hook
        (`lightning_base.on_train_epoch_end`) — so at callback epoch-end the log
        often holds nothing but a fresh reset. By the next epoch's start those
        evaluation episodes have completed, so the log holds a whole one.
        """
        self.write()
