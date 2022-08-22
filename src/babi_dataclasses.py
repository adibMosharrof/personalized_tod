import abc
from dataclasses import dataclass
from pathlib import Path
from dotmap import DotMap
from tqdm import tqdm


@dataclass
class BaseTask(abc.ABC):
    turn_id: int
    user: str
    system: str
    dialog_id: int

    @abc.abstractmethod
    def get_dataset(self, file_path: Path, split_percent: float) -> any:
        raise (NotImplementedError())


@dataclass
class BabiTask1(BaseTask):
    @classmethod
    def from_raw_line(self, line: str, dialog_id) -> "BabiTask1":
        id_user, system = line.strip().split("\t")
        id_sep_index = id_user.index(" ")
        turn_id = int(id_user[:id_sep_index])
        user = id_user[id_sep_index + 1 :]
        return self(turn_id, user, system, dialog_id)

    @classmethod
    def get_dataset(self, file_path: Path, split_percent: float) -> DotMap:
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
                data.append(BabiTask1.from_raw_line(line, dialog_id))
                i += 1
            dialog_index.pop()
            num_dialogs = int(len(dialog_index) * split_percent) - 1
            data = data[: dialog_index[num_dialogs]]
            return DotMap(data=data, dialog_index=dialog_index)


BabiTask1SystemColumns = DotMap(
    task="task",
    cuisine="cuisine",
    location="location",
    party_size="party_size",
    price_range="price_range",
)
