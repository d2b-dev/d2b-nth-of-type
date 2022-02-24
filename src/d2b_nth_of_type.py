from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import TypeVar

from d2b.hookspecs import hookimpl


__version__ = "1.0.1"


class Defaults:
    ENABLED = True
    SORT_BY = "SeriesNumber:asc"
    GROUP_BY = "SeriesDescription"


D2B_NTH_OF_TYPE_ENABLED = "D2B_NTH_OF_TYPE_ENABLED"
D2B_NTH_OF_TYPE_SORT_BY = "D2B_NTH_OF_TYPE_SORT_BY"
D2B_NTH_OF_TYPE_GROUP_BY = "D2B_NTH_OF_TYPE_GROUP_BY"


@hookimpl
def prepare_run_parser(optional: argparse._ArgumentGroup) -> None:
    add_arguments(optional)


def add_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup):
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument(
        "--nth-of-type-enabled",
        dest="nth_of_type_enabled",
        action="store_true",
        default=_is_yes(os.getenv(D2B_NTH_OF_TYPE_ENABLED, Defaults.ENABLED)),
        help="Enable the 'd2b-nth-of-type' plugin (default). "
        "If this flag is not set, then the program will use the value "
        f"from the environment variable {D2B_NTH_OF_TYPE_ENABLED}",
    )
    enabled_group.add_argument(
        "--nth-of-type-disabled",
        dest="nth_of_type_enabled",
        action="store_false",
        help="Disable the 'd2b-nth-of-type' plugin. "
        "If this flag is not set, then the program will use the value "
        f"from the environment variable {D2B_NTH_OF_TYPE_ENABLED}",
    )
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
    enabled: bool = options["nth_of_type_enabled"]

    if enabled:
        nth_of_type(files, sortby=sortby, groupby=groupby)


def nth_of_type(files: list[Path], sortby: str, groupby: str):
    """Modifies sidecars in-place"""
    # filter non-JSON files
    sidecar_files = filter_files(files)

    # load the json files into Sidecar objects
    sidecars = load_sidecars(sidecar_files)

    # order the sidecars
    sort_prop, reverse = parse_sortby(sortby)
    sort_config = SortConfig.infer_from_sidecars(sidecars, sort_prop, reverse)
    ordered_sidecars = sort_sidecars(sidecars, sort_config)

    # group the sidecars
    _group_keys = parse_groupby(groupby)
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


def sort_sidecars(sidecars: list[Sidecar], sort_config: SortConfig) -> list[Sidecar]:
    path_sorted = sorted(sidecars, key=lambda s: s.path)
    key_sorted = sorted(path_sorted, key=sort_config.key, reverse=sort_config.reverse)
    return key_sorted


class SortConfig(NamedTuple):
    key: Callable[[Sidecar], Any]
    reverse: bool

    @classmethod
    def infer_from_sidecars(
        cls,
        sidecars: list[Sidecar],
        prop: str,
        reverse: bool,
    ) -> SortConfig:
        if any(isinstance(sidecar.data.get(prop), int) for sidecar in sidecars):
            # coerce everything to INT
            key_fn = key_fn_factory(prop, default=sys.maxsize, converter=int)
        elif any(isinstance(sidecar.data.get(prop), float) for sidecar in sidecars):
            # coerce everything to FLOAT
            key_fn = key_fn_factory(prop, default=float(sys.maxsize), converter=float)
        else:
            # coerce everything to STR
            key_fn = key_fn_factory(prop, default="", converter=str)

        return cls(key_fn, reverse)


T = TypeVar("T", int, float, str)


def key_fn_factory(
    prop: str,
    /,
    *,
    default: T,
    converter: Callable[[Any], T],
) -> Callable[[Sidecar], tuple[T, str]]:
    def key_fn(sidecar: Sidecar) -> tuple[T, str]:
        try:
            return (converter(sidecar.data[prop]), str(sidecar.path))
        except (TypeError, KeyError):
            return (default, str(sidecar.path))

    return key_fn


def parse_sortby(s: str) -> tuple[str, bool]:
    prop, _, direction = s.partition(":")
    reverse = direction == "desc"
    return (prop, reverse)


def parse_groupby(s: str) -> tuple[str]:
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


def _is_yes(v: Any):
    return str(v).strip() not in ("0", "false", "False", "n", "no", "N", "No" "NO")
