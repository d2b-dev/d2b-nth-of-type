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
from d2b_nth_of_type import parse_group_keys
from d2b_nth_of_type import parse_sort_key
from d2b_nth_of_type import Sidecar
from d2b_nth_of_type import sort_sidecars


class SidecarData(NamedTuple):
    filename: str
    series_number: int
    series_description: str
    phase_encoding_direction: str


@pytest.fixture
def fake_sidecar_files(tmpdir: str) -> list[Path]:
    sidecar_data = [
        SidecarData("d.json", 2, "desc2", "k"),
        SidecarData("b.json", 1, "desc3", "j"),
        SidecarData("c.json", 3, "desc2", "i"),
        SidecarData("a.json", 1, "desc1", "i"),
    ]

    files: list[Path] = []
    for scd in sidecar_data:
        fp = Path(tmpdir) / scd.filename
        data = {
            "SeriesNumber": scd.series_number,
            "SeriesDescription": scd.series_description,
            "PhaseEncodingDirection": scd.phase_encoding_direction,
        }
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
        assert "SeriesNumber" in sidecar.data
        assert "SeriesDescription" in sidecar.data
        assert "PhaseEncodingDirection" in sidecar.data


def test_sort_sidecars_asc(fake_sidecars: list[Sidecar]):
    sort_key = parse_sort_key("SeriesNumber:asc")
    ordered = sort_sidecars(fake_sidecars, sort_key=sort_key)

    first = ordered[0]
    last = ordered[-1]

    assert first.path.name == "a.json"
    assert first.data["SeriesNumber"] == 1
    assert last.path.name == "c.json"
    assert last.data["SeriesNumber"] == 3


def test_sort_sidecars_desc(fake_sidecars: list[Sidecar]):
    sort_key = parse_sort_key("SeriesDescription:desc")
    ordered = sort_sidecars(fake_sidecars, sort_key=sort_key)

    first = ordered[0]
    last = ordered[-1]

    assert first.path.name == "b.json"
    assert first.data["SeriesDescription"] == "desc3"
    assert last.path.name == "a.json"
    assert last.data["SeriesDescription"] == "desc1"


def test_group_sidecars_single_key(fake_sidecars: list[Sidecar]):
    sort_key = parse_sort_key("SeriesNumber:asc")
    ordered = sort_sidecars(fake_sidecars, sort_key)

    group_keys = parse_group_keys("SeriesDescription")
    grouped = group_sidecars(ordered, group_keys=group_keys)

    assert [sc.path.name for sc in grouped[("desc1",)]] == ["a.json"]
    assert [sc.path.name for sc in grouped[("desc2",)]] == ["d.json", "c.json"]


def test_group_sidecars_multiple_keys(fake_sidecars: list[Sidecar]):
    sort_key = parse_sort_key("SeriesNumber:asc")
    ordered = sort_sidecars(fake_sidecars, sort_key)

    group_keys = parse_group_keys("SeriesDescription,,SeriesNumber")
    grouped = group_sidecars(ordered, group_keys=group_keys)

    # notice that the empty group key is dropped and that each group key
    # is a string despite SeriesNumber (i.e. the second element in the
    # tuple) is an integer in the sidecar
    assert [sc.path.name for sc in grouped[("desc1", "1")]] == ["a.json"]
    assert [sc.path.name for sc in grouped[("desc2", "3")]] == ["c.json"]
