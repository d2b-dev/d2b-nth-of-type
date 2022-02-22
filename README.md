# d2b-nth-of-type

[![PyPI Version](https://img.shields.io/pypi/v/d2b-nth-of-type.svg)](https://pypi.org/project/d2b-nth-of-type/) [![Code Style](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/lint.yaml/badge.svg)](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/lint.yaml) [![Type Check](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/type-check.yaml/badge.svg)](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/type-check.yaml) [![Tests](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/test.yaml/badge.svg)](https://github.com/d2b-dev/d2b-nth-of-type/actions/workflows/test.yaml)

Plugin for the d2b package to deterministically uniquify different acquisition runs

## Intallation

```bash
pip install d2b-nth-of-type
```

## Getting Started

This plugin will inject properties into each JSON sidecar which distinguish different runs from one another, which can subsequently be used in `criteria` specifications in d2b config files.

Specifically, this plugin will inject a property, `__nth_of_type__`, into each JSON sidecar. The value of this property is a 0-indexed integer.

**This plugin groups acquisitions and gives each run in the group a unique (0-indexed) label, incrementing in the specified order.**

By default, acquisitions are grouped by `SeriesDescription` and are ordered by `SeriesNumber` (in _ascending_ order).

For example, if a subject has a session in which 3 fieldmaps are acquired, and say, for example, the 3 sidecars (truncated) are of the form:

- fieldmap 1:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 3
  }
  ```

- fieldmap 2:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 16
  }
  ```

- fieldmap 3:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 24
  }
  ```

Then, when running `d2b run` with this plugin installed, this plugin will inject `__nth_of_type__` into each sidecar, resulting in sidecars (truncated) which look like:

- fieldmap 1:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 3,
    "__nth_of_type__": 0
  }
  ```

- fieldmap 2:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 16,
    "__nth_of_type__": 1
  }
  ```

- fieldmap 3:

  ```json
  {
    "SeriesDescription": "my_fmap",
    "SeriesNumber": 24,
    "__nth_of_type__": 2
  }
  ```

The utility is that these values will be invariant under changes to `SeriesNumber`.

So, if the "first" fmap is intended for a BOLD acquisition, and the "second" fmap is intended for a PCASL acquisition, we can safely pick-out each fmap using the `__nth_of_type__` field.

For example, we could have a d2b configuration file of the following form (_NOTE: [`d2b-yaml`](https://github.com/d2b-dev/d2b-yaml) plugin required for yaml config files, and you probably also want the [`d2b-asl`](https://github.com/d2b-dev/d2b-asl) plugin for ASL data_):

```yaml
descriptions:
  # BOLD Resting State
  - id: my-bold-rs
    dataType: func
    modalityLabel: bold
    customLabels: task-rest

    criteria:
      # ...

  # PCASL
  - id: my-pcasl
    dataType: perf
    modalityLabel: asl
    customLabels:
      acq: pcasl

    criteria:
      # ...
    aslContext:
      # ...

  # FIRST FMAP - for BOLD - phase encoding = AP
  - dataType: fmap
    modalityLabel: epi
    customLabels:
      dir: AP
    IntendedFor:
      - my-bold-rs
    criteria:
      ManufacturersModelName: Prisma_fit
      SidecarFilename: "*SpinEchoFieldMap_AP*"
      __nth_of_type__: 0 # <-- HERE

  # FIRST FMAP - for BOLD - phase encoding = PA
  - dataType: fmap
    modalityLabel: epi
    customLabels:
      dir: PA
    IntendedFor:
      - my-bold-rs
    criteria:
      ManufacturersModelName: Prisma_fit
      SidecarFilename: "*SpinEchoFieldMap_PA*"
      __nth_of_type__: 0 # <-- HERE

  # SECOND FMAP - for PCASL - phase encoding = AP
  - dataType: fmap
    modalityLabel: epi
    customLabels:
      dir: AP
    IntendedFor:
      - my-pcasl
    criteria:
      ManufacturersModelName: Prisma_fit
      SidecarFilename: "*SpinEchoFieldMap_AP*"
      __nth_of_type__: 1 # <-- HERE

  # SECOND FMAP - for PCASL - phase encoding = PA
  - dataType: fmap
    modalityLabel: epi
    customLabels:
      dir: PA
    IntendedFor:
      - my-pcasl
    criteria:
      ManufacturersModelName: Prisma_fit
      SidecarFilename: "*SpinEchoFieldMap_PA*"
      __nth_of_type__: 1 # <-- HERE
```

## Configuration

This plugin can be configured by via options to the `d2b run` command or via the environment:

- `--nth-of-type-group-by <string>`

  Which sidecar field(s) to group acquisitions by (default: `SeriesDescription`).

  To group by more then one field pass a comma separated list, ex: `SeriesDescription,RepetitionTime`

  Can also be configured via the `D2B_NTH_OF_TYPE_GROUP_BY` environment variable. The command line argument takes precedence over the environment variable.

- `--nth-of-type-sort-by <string>`

  Which sidecar field to sort the acquisitions by (default: `SeriesNumber:asc`).

  To sort in descending order append `:desc` onto the fieldname, ex: `SeriesNumber:desc`. If no direction/ordering suffix (`:asc`/`:desc`) is present then the acquisitions are sorted in ascending order.

  Can also be configured via the `D2B_NTH_OF_TYPE_SORT_BY` environment variable. The command line argument takes precedence over the environment variable.

## Contributing

1. Have or install a recent version of `poetry` (version >= 1.1)
1. Fork the repo
1. Setup a virtual environment (however you prefer)
1. Run `poetry install`
1. Run `pre-commit install`
1. Add your changes (adding/updating tests is always nice too)
1. Commit your changes + push to your fork
1. Open a PR
