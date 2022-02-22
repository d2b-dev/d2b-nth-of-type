from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

from d2b.hookspecs import hookimpl


__version__ = "0.1.0"


class Defaults:
    SORT_BY = "SeriesNumber:asc"
    GROUP_BY = "SeriesDescription"


D2B_NTH_OF_TYPE_SORT_BY = "D2B_NTH_OF_TYPE_SORT_BY"
D2B_NTH_OF_TYPE_GROUP_BY = "D2B_NTH_OF_TYPE_GROUP_BY"


@hookimpl
def prepare_run_parser(optional: argparse._ArgumentGroup) -> None:
    add_arguments(optional)


def add_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup):
    parser.add_argument(
        "--nth-of-type-sort-by",
        type=str,
        default=os.getenv(D2B_NTH_OF_TYPE_SORT_BY, Defaults.SORT_BY),
        help="A property in the JSON sidecar used to determine how files are "
        "ordered. If this flag is not set, then the program will use the value "
        f"from the environment variable {D2B_NTH_OF_TYPE_SORT_BY}, if neither "
        "the flag, nor enviroment variable are set then the default is used "
        "(%(default)s). By default files are sorted in ascending order, "
        "to sort in descending order, add ':desc' to the key, for example, "
        "to sort by descending SeriesNumber (i.e. largest first) then this "
        "parameter should be 'SeriesNumber:desc'. Sidecar files which do not "
        "have the given key are sorted in ascending order using their file "
        "path. (default: %(default)s)",
    )
    parser.add_argument(
        "--nth-of-type-group-by",
        type=str,
        default=os.getenv(D2B_NTH_OF_TYPE_GROUP_BY, Defaults.GROUP_BY),
        help="Property(ies) in the JSON sidecar used to determine how files are "
        "grouped. If this flag is not set, then the program will use the value "
        f"from the environment variable {D2B_NTH_OF_TYPE_GROUP_BY}, if neither "
        "the flag, nor enviroment variable are set then the default is used "
        "(%(default)s). This should be a comma-separated-list of JSON "
        "sidecar properties. (default: %(default)s)",
    )


@hookimpl
def pre_run_logs(logger: logging.Logger) -> None:
    logger.info(f"d2b-nth-of-type:version: {__version__}")


@hookimpl
def prepare_collected_files(files: list[Path], options: dict[str, Any]) -> None:
    """Provide files to consider for description <-> file matching"""
    sortby: str = options["nth_of_type_sort_by"]
    groupby: str = options["nth_of_type_group_by"]

    nth_of_type(files, sortby=sortby, groupby=groupby)


def nth_of_type(files: list[Path], sortby: str, groupby: str):
    """Modifies sidecars in-place"""
    # filter non-JSON files
    sidecar_files = filter_files(files)

    # load the json files into Sidecar objects
    sidecars = load_sidecars(sidecar_files)

    # order the sidecars
    _sort_key = parse_sort_key(sortby)
    ordered_sidecars = sort_sidecars(sidecars, _sort_key)

    # group the sidecars
    _group_keys = parse_group_keys(groupby)
    grouped_sidcars = group_sidecars(ordered_sidecars, _group_keys)

    # rewrite the sidecars with the changes
    rewrite_files(grouped_sidcars, sortby=sortby, groupby=groupby)


def filter_files(files: list[Path]) -> list[Path]:
    return [fp for fp in files if fp.suffix == ".json"]


class Sidecar(NamedTuple):
    path: Path
    data: dict[str, Any]


def load_sidecars(files: list[Path]) -> list[Sidecar]:
    return [Sidecar(fp, json.loads(fp.read_text())) for fp in files]


def sort_sidecars(sidecars: list[Sidecar], sort_key: SortKey) -> list[Sidecar]:
    path_sorted = sorted(sidecars, key=lambda s: s.path)
    key_sorted = sorted(path_sorted, key=sort_key.key, reverse=sort_key.reverse)
    return key_sorted


class SortKey(NamedTuple):
    key: Callable[[Sidecar], Any]
    reverse: bool


def parse_sort_key(s: str) -> SortKey:
    prop, _, direction = s.partition(":")
    reverse = direction == "desc"

    def key(sidecar: Sidecar) -> Any:
        return sidecar.data.get(prop)

    return SortKey(key, reverse)


def parse_group_keys(s: str) -> tuple[str]:
    return tuple(key.strip() for key in s.split(",") if key.strip())


GroupedSidecars = Dict[Tuple[str, ...], List[Sidecar]]


def group_sidecars(sidecars: list[Sidecar], group_keys: tuple[str]) -> GroupedSidecars:
    grouped_sidecars: GroupedSidecars = defaultdict(list)
    for sidecar in sidecars:
        group = tuple(str(sidecar.data.get(gk)) for gk in group_keys)
        grouped_sidecars[group].append(sidecar)

    return grouped_sidecars


def rewrite_files(grouped_sidcars: GroupedSidecars, sortby: str, groupby: str) -> None:
    for sidecars in grouped_sidcars.values():
        for i, sidecar in enumerate(sidecars):
            sidecar.data["__nth_of_type__"] = i
            sidecar.data["__nth_of_type_sortby__"] = sortby
            sidecar.data["__nth_of_type_groupby__"] = groupby

            sidecar_s = json.dumps(sidecar.data, indent=2)

            sidecar.path.write_text(sidecar_s)
