from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import pytest

from d2b_nth_of_type import add_arguments
from d2b_nth_of_type import filter_files
from d2b_nth_of_type import group_sidecars
from d2b_nth_of_type import load_sidecars
from d2b_nth_of_type import parse_groupby
from d2b_nth_of_type import parse_sortby
from d2b_nth_of_type import Sidecar
from d2b_nth_of_type import sort_sidecars
from d2b_nth_of_type import SortConfig


class SidecarData(NamedTuple):
    filename: str
    series_number: int | str | None
    series_description: str
    phase_encoding_direction: str


@pytest.fixture
def fake_sidecar_files(tmpdir: str) -> list[Path]:
    sidecar_data = [
        SidecarData("d.json", 2, "desc2", "k"),
        SidecarData("b.json", 1, "desc3", "j"),
        SidecarData("c.json", 3, "desc2", "i"),
        SidecarData("a.json", 1, "desc1", "i"),
        SidecarData("f.json", None, "desc3", "k"),
        SidecarData("e.json", "13", "desc2", "j"),
    ]

    files: list[Path] = []
    for scd in sidecar_data:
        data = {}
        if scd.series_number is not None:
            data["SeriesNumber"] = scd.series_number
        data["SeriesDescription"] = scd.series_description
        data["PhaseEncodingDirection"] = scd.phase_encoding_direction

        fp = Path(tmpdir) / scd.filename
        fp.write_text(json.dumps(data))
        files.append(fp)

    return files


@pytest.fixture
def fake_sidecars(fake_sidecar_files: list[Path]) -> list[Sidecar]:
    return load_sidecars(fake_sidecar_files)


def test_add_arguments():
    parser = argparse.ArgumentParser()

    add_arguments(parser)

    args = parser.parse_args([])
    assert args.nth_of_type_sort_by == "SeriesNumber:asc"
    assert args.nth_of_type_group_by == "SeriesDescription"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ([], []),
        ([Path("a.json"), Path("b/c.json")], [Path("a.json"), Path("b/c.json")]),
        (
            [Path("a.json"), Path("b/c.json"), Path("d.nii.gz")],
            [Path("a.json"), Path("b/c.json")],
        ),
        ([Path("a.txt"), Path("b/c.py"), Path("d.nii.gz")], []),
    ],
)
def test_filter_files(files: list[Path], expected: list[Path]):
    assert filter_files(files) == expected


def test_load_sidecars(fake_sidecar_files: list[Path]):
    sidecars = load_sidecars(fake_sidecar_files)

    for sidecar, original_fp in zip(sidecars, fake_sidecar_files):
        assert sidecar.path == original_fp
        assert "SeriesDescription" in sidecar.data
        assert "PhaseEncodingDirection" in sidecar.data


def test_sort_sidecars_asc(fake_sidecars: list[Sidecar]):
    prop, reverse = parse_sortby("SeriesNumber:asc")
    sort_config = SortConfig.infer_from_sidecars(fake_sidecars, prop, reverse)
    ordered = sort_sidecars(fake_sidecars, sort_config=sort_config)

    first = ordered[0]
    last = ordered[-1]

    assert first.path.name == "a.json"
    assert first.data["SeriesNumber"] == 1
    assert last.path.name == "f.json"


def test_sort_sidecars_desc(fake_sidecars: list[Sidecar]):
    prop, reverse = parse_sortby("SeriesDescription:desc")
    sort_config = SortConfig.infer_from_sidecars(fake_sidecars, prop, reverse)
    ordered = sort_sidecars(fake_sidecars, sort_config=sort_config)

    first = ordered[0]
    last = ordered[-1]

    assert first.path.name == "f.json"
    assert first.data["SeriesDescription"] == "desc3"
    assert last.path.name == "a.json"
    assert last.data["SeriesDescription"] == "desc1"


def test_group_sidecars_single_key(fake_sidecars: list[Sidecar]):
    prop, reverse = parse_sortby("SeriesNumber:asc")
    sort_config = SortConfig.infer_from_sidecars(fake_sidecars, prop, reverse)
    ordered = sort_sidecars(fake_sidecars, sort_config=sort_config)

    group_keys = parse_groupby("SeriesDescription")
    grouped = group_sidecars(ordered, group_keys=group_keys)

    assert [sc.path.name for sc in grouped[("desc1",)]] == ["a.json"]
    assert [sc.path.name for sc in grouped[("desc2",)]] == [
        "d.json",
        "c.json",
        "e.json",
    ]


def test_group_sidecars_multiple_keys(fake_sidecars: list[Sidecar]):
    prop, reverse = parse_sortby("SeriesNumber:asc")
    sort_config = SortConfig.infer_from_sidecars(fake_sidecars, prop, reverse)
    ordered = sort_sidecars(fake_sidecars, sort_config=sort_config)

    group_keys = parse_groupby("SeriesDescription,,SeriesNumber")
    grouped = group_sidecars(ordered, group_keys=group_keys)

    # notice that the empty group key is dropped and that each group key
    # is a string despite SeriesNumber (i.e. the second element in the
    # tuple) is an integer in the sidecar
    assert [sc.path.name for sc in grouped[("desc1", "1")]] == ["a.json"]
    assert [sc.path.name for sc in grouped[("desc2", "3")]] == ["c.json"]
