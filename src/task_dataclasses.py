import abc
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from dotmap import DotMap
from tqdm import tqdm

from tod_utils import SpecialPredictions, SpecialTokens


class TaskConstants(str, Enum):
    API_CALL = "api_call"
    OPTIONS = "what do you think of this option:"


@dataclass
class BaseTask(abc.ABC):
    turn_id: int = None
    user: str = None
    system: str = None
    dialog_id: int = None
    query: str = None
    slots: list[str] = field(default_factory=list)
    option: str = None

    @abc.abstractclassmethod
    def from_raw_line(self, line: str, dialog_id: int) -> Optional["BaseTask"]:
        raise (NotImplementedError())

    @classmethod
    def read_raw_data(self, file_path: Path, split_percent: float) -> DotMap:
        with open(file_path, "r") as f:
            dialog_id = 1
            data = []
            turn_index: int = 0
            dialog_index: list[int] = [0]
            for line in tqdm(f.readlines()):
                if line == "\n":
                    dialog_id += 1
                    dialog_index.append(turn_index)
                    continue
                processed_row = self.from_raw_line(line.strip(), dialog_id)
                if processed_row:
                    data.append(processed_row)
                    turn_index += 1
            dialog_index.pop()
            num_dialogs = int(len(dialog_index) * split_percent) - 1
            data = data[: dialog_index[num_dialogs]]
            return DotMap(data=data, dialog_index=dialog_index)

    @abc.abstractclassmethod
    def dummy(self) -> "BaseTask":
        raise (NotImplementedError())


@dataclass
class KnowledgeBaseLine:
    name: str
    col_name: str
    value: str

    def __init__(self, line: str):
        _, name, col_name, value = line.split(" ")
        self.col_name = col_name.split("_")[1]
        self.value = value
        self.name = name


@dataclass
class KnowledgeBase:
    name: str = None
    phone: str = None
    cuisine: str = None
    address: str = None
    location: str = None
    number: str = None
    price: str = None
    rating: int = None

    def update_col_from_line(self, line):
        kb_line = KnowledgeBaseLine(line)
        self.name = kb_line.name
        setattr(self, kb_line.col_name, kb_line.value)


@dataclass
class BabiTask12(BaseTask):
    @classmethod
    def from_raw_line(self, line: str, dialog_id) -> "BabiTask12":
        id_user, system = line.split("\t")
        id_sep_index = id_user.index(" ")
        turn_id = int(id_user[:id_sep_index])
        user = id_user[id_sep_index + 1 :]
        if system.startswith(TaskConstants.API_CALL):
            return self(
                query=TaskConstants.API_CALL.value,
                slots=system.split(" ")[1:],
                dialog_id=dialog_id,
                turn_id=turn_id,
                user=user,
            )
        return self(turn_id=turn_id, user=user, system=system, dialog_id=dialog_id)

    @classmethod
    def dummy(self) -> "BabiTask12":
        return self(
            turn_id=-1,
            user=SpecialPredictions.DUMMY,
            system=SpecialPredictions.DUMMY,
            dialog_id=-1,
        )


@dataclass
class BabiTask3(BaseTask):
    knowledge_base: list[KnowledgeBase] = field(default_factory=list)

    @classmethod
    def from_raw_line(
        self, line: str, dialog_id: int, turn_id: int, kb: list[KnowledgeBase]
    ) -> "BabiTask3":

        option = None
        _, utterances = line.split(" ")
        user, system = utterances.split("\t")
        if TaskConstants.OPTIONS in system:
            option = system.split(" ")[-1]

        return self(
            turn_id=turn_id,
            user=user,
            system=system,
            dialog_id=dialog_id,
            knowledge_base=kb,
            option=option,
        )

    @classmethod
    def get_dialog_from_line(self, line: str) -> str:
        return line.split("\t")[2]

    @classmethod
    def read_raw_data(self, file_path: Path, split_percent: float) -> DotMap:
        with open(file_path, "r") as f:
            dialog_id = 1
            data = []
            turn_index: int = 0
            dialog_index: list[int] = [0]
            task_ref = self(dialog_id=dialog_id)

            current_kb = KnowledgeBase()
            turn_kb: list[KnowledgeBase] = []
            for line in tqdm(f.readlines()):
                if line == "\n":
                    dialog_id += 1
                    dialog_index.append(turn_index)
                    task_ref = self(dialog_id=dialog_id)
                    turn_kb = []
                    current_kb = KnowledgeBase()
                    continue
                line = line.strip()
                if "\t" in line:
                    processed_row = self.from_raw_line(
                        line, dialog_id, turn_id, task_ref, turn_kb
                    )
                    if processed_row:
                        data.append(processed_row)
                        turn_index += 1
                else:
                    current_kb.update_col_from_line(line)
                    if current_kb.rating:
                        turn_kb.append(current_kb)
                        current_kb = KnowledgeBase()

            dialog_index.pop()
            num_dialogs = int(len(dialog_index) * split_percent) - 1
            data = data[: dialog_index[num_dialogs]]
            return DotMap(data=data, dialog_index=dialog_index)

    @classmethod
    def dummy(self) -> "BaseTask":
        raise (NotImplementedError())


@dataclass
class PersonalizedTask1(BaseTask):
    gender: str = None
    age_group: str = None

    @classmethod
    def from_raw_line(self, line: str, dialog_id) -> Optional["PersonalizedTask1"]:
        if line.startswith("1"):
            _, self.gender, self.age_group = line.split(" ")
            return None
        id_user, system = line.split("\t")
        id_sep_index = id_user.index(" ")
        turn_id = int(id_user[:id_sep_index])
        user = id_user[id_sep_index + 1 :]
        return self(
            turn_id=turn_id,
            user=user,
            system=system,
            dialog_id=dialog_id,
            gender=self.gender,
            age_group=self.age_group,
        )


Task1_System_Columns = DotMap(
    task="task",
    cuisine="cuisine",
    location="location",
    party_size="party_size",
    price_range="price_range",
)
