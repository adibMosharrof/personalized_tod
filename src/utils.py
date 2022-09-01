import csv
import logging
from pathlib import Path
from dataclass_csv import DataclassReader


def write_csv(headers: list[str], data, file_name: Path):
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


def read_csv_dataclass(path: str, d_class):
    with open(path) as f:
        reader = DataclassReader(f, d_class)
        return [r for r in reader]


def get_logger():
    return logging.getLogger(__name__)
