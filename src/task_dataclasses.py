import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from dotmap import DotMap
from tqdm import tqdm


class QueryEnum(str, Enum):
    API_CALL = "api_call"


@dataclass
class BaseTask(abc.ABC):
    turn_id: int = None
    user: str = None
    system: str = None
    dialog_id: int = None
    query: str = None
    slots: list[str] = field(default_factory=list)

    @abc.abstractclassmethod
    def from_raw_line(self, line: str, dialog_id: int) -> Optional["BaseTask"]:
        raise (NotImplementedError())

    @classmethod
    def read_raw_data(self, file_path: Path, split_percent: float) -> DotMap:
        with open(file_path, "r") as f:
            dialog_id = 1
            data = []
            i: int = 0
            dialog_index: list[int] = [0]
            for line in tqdm(f.readlines()):
                if line == "\n":
                    dialog_id += 1
                    dialog_index.append(i)
                    continue
                processed_row = self.from_raw_line(line.strip(), dialog_id)
                if processed_row:
                    data.append(processed_row)
                    i += 1
            dialog_index.pop()
            num_dialogs = int(len(dialog_index) * split_percent) - 1
            data = data[: dialog_index[num_dialogs]]
            return DotMap(data=data, dialog_index=dialog_index)


@dataclass
class BabiTask1(BaseTask):
    @classmethod
    def from_raw_line(self, line: str, dialog_id) -> "BabiTask1":
        id_user, system = line.split("\t")
        id_sep_index = id_user.index(" ")
        turn_id = int(id_user[:id_sep_index])
        user = id_user[id_sep_index + 1 :]
        if system.startswith(QueryEnum.API_CALL):
            return self(
                query=QueryEnum.API_CALL.value,
                slots=system.split(" ")[1:],
                dialog_id=dialog_id,
                turn_id=turn_id,
                user=user,
            )
        return self(turn_id=turn_id, user=user, system=system, dialog_id=dialog_id)


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
