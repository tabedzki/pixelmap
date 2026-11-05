# Contributing / Developer Guide

This guide is for contributors and developers who want to work on PixelMap.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/m-beau/pixelmap.git
cd pixelmap

# Install with dev dependencies
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Running Locally

```bash
# Launch the GUI
uv run pixelmap

# Or if installed with pip
pixelmap
```

## Architecture Overview

```
pixelmap/
├── __init__.py          # Public API exports
├── backend.py           # Core electrode selection logic
├── constants.py         # Probe configurations and presets
├── types.py             # Data classes (Electrode, etc.)
├── gui/
│   ├── gui.py           # Panel frontend
│   ├── app_util.py      # GUI helper functions
│   └── assets/          # Images and static files
├── utils/
│   └── imro.py          # IMRO file reading/writing
└── wiring_maps/         # CSV/JSON files defining electrode-ADC wiring
```

### Key Modules

**`backend.py`**
: Core logic for electrode selection and validation. Handles ADC wiring constraints via `find_forbidden_electrodes()` and generates valid channel configurations.

**`gui/gui.py`**
: Browser-based interface built with [NiceGUI](https://nicegui.io/). Handles user interactions, electrode visualization, and IMRO file downloads.

**`utils/imro.py`**
: Functions for reading, writing, and parsing IMRO files. The `generate_imro_channelmap()` function is the main entry point for programmatic use.

**`constants.py`**
: Probe group definitions, electrode counts, and preset configurations. Edit this when adding new probe types or presets.

**`types.py`**
: Data classes used throughout the codebase, including the `Electrode` class.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pixelmap

# Run specific test file
pytest tests/test_backend.py
```

### Writing Tests

Tests are in the `tests/` directory. When adding new functionality:

1. Add tests for the new feature
2. Ensure existing tests still pass
3. Aim for coverage of edge cases

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Adding New Probe Types

To add support for a new probe (e.g., Quadbase, NXT):

1. **Add wiring map**: Create `wiring_maps/<probe_type>_wiring.csv` defining electrode-to-ADC mappings

2. **Update constants**: In `constants.py`, add entries to:
   - `PROBE_TYPE_MAP` - Probe groups and supported part numbers
   - `PROBE_FEATURES` - Per-part-number electrode, shank and readout-channel counts
   - `SUPPORTED_*_PRESETS` - Available presets (if applicable)

3. **Update GUI**: In `gui/gui.py`, add the probe to the dropdown options

4. **Add tests**: Create tests for the new probe type

5. **Update docs**: Add to the supported probes table in `index.md` and `README.md`

## Contributing

To contribute:

1. Fork the repository
2. Create a branch with a descriptive name (e.g., `feature/your-feature-name` or `fix/bug-description`)
3. Make your changes and run tests: `pytest`
4. Push to your fork and open a pull request **to the `main` branch**

See [CONTRIBUTING.md](https://github.com/m-beau/pixelmap/blob/main/CONTRIBUTING.md) for full guidelines on reporting bugs, requesting features, and code style.

## Releases

`main` is the only long-lived branch, and it holds unreleased code. A release is a git tag: the version comes from tags through setuptools-scm, so there is no version number to bump in the code.

| You push | Python version | Docker image tags |
|---|---|---|
| A merge to `main` | `1.6.0.dev4+gabc1234` | `edge`, `sha-abc1234` |
| Tag `v1.6.0rc1` | `1.6.0rc1` | `v1.6.0rc1`, `1.6.0rc1` |
| Tag `v1.6.0` | `1.6.0` | `v1.6.0`, `1.6.0`, `1.6`, `latest` |

Production follows `latest` through Watchtower, so **pushing a final release tag deploys it**. Tests must pass before any image is published.

To cut a release:

1. Update the version and date in `CITATION.cff` in a PR, and merge it.
2. Optionally tag a release candidate and try its image.
3. Tag the commit and push the tag: `git tag v1.6.0 && git push upstream v1.6.0`.

Tags must look like `vX.Y.Z`, optionally followed by `aN`, `bN` or `rcN`, and `.devN`. The image build fails on anything else.

To fix a release while `main` has changes you don't want to ship yet, branch from its tag, pick the fix onto that branch, and tag it:

```bash
git switch -c release/1.5.x v1.5.0
git cherry-pick <commit>
git tag v1.5.1
```

Only the highest final release gets `latest`, so `v1.5.1` won't replace `v1.6.0` in production.

To roll back, pin the image in `docker-compose.yml`, for example `ghcr.io/m-beau/pixelmap:v1.4.3`.
