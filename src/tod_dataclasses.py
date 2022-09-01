from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

from transformers import AutoTokenizer, GPT2LMHeadModel
from task_dataclasses import BaseTask
from tod_utils import SpecialTokens


@dataclass
class TodContext:
    turns: list[BaseTask] = field(default_factory=list)
    current_user_utterance: str = field(default_factory=str)

    def __str__(self) -> str:
        out = SpecialTokens.begin_context
        for turn in self.turns:
            if turn.user:
                out += SpecialTokens.user + turn.user
            if turn.system:
                out += SpecialTokens.system + turn.system

        out += (
            SpecialTokens.begin_last_user_utterance
            + self.current_user_utterance
            + SpecialTokens.end_last_user_utterance
        )
        out += SpecialTokens.end_context
        return out


@dataclass
class TodTarget:
    turn: BaseTask = field(default_factory=BaseTask)

    def _get_content(self) -> str:
        if not self.turn.query:
            return self.turn.system
        out = SpecialTokens.begin_query + self.turn.query + SpecialTokens.end_query
        out += (
            SpecialTokens.begin_slots
            + " ".join(self.turn.slots)
            + SpecialTokens.end_slots
        )
        return out

    def __str__(self) -> str:
        return (
            SpecialTokens.begin_target + self._get_content() + SpecialTokens.end_target
        )


@dataclass
class TodTurnCsvRow:
    context: str
    target: str
    dialog_id: str
    turn_id: str

    def __init__(
        self, dialog_id: int, turn_id: int, context: TodContext, target: TodTarget
    ):
        self.dialog_id = str(dialog_id)
        self.turn_id = str(turn_id)
        self.context = str(context)
        self.target = str(target)


@dataclass
class TodTurn:
    context: TodContext
    target: TodTarget
    dialog_id: int
    turn_id: int

    def to_csv_row(self) -> list[str]:
        return [
            self.dialog_id,
            self.turn_id,
            str(self.context),
            str(self.target),
        ]


TodDatasetRow = namedtuple("TodDatasetRow", "context target")
TodTestDatasetBatch = namedtuple(
    "TodTestDatasetBatch",
    "context_tokens context_attention_masks target_tokens target_attention_masks contexts targets",
)


def get_csv_data_path(
    dataset_name: str,
    task_name: str,
    step: str,
    split_percent: float,
    out_root: Path,
) -> Path:
    step_dir = out_root / step
    return (
        step_dir
        / f'{"_".join([dataset_name, task_name, step, str(split_percent)])}.csv'
    )
