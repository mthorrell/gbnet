# Conda distribution

The recipe in `recipe/meta.yaml` builds GBNet with its native dependencies from
conda-forge. It does not install pip wheels for those dependencies. The package
is pure Python, but the recipe builds per platform so it can retain the macOS and
Intel-specific dependency bounds in the Python metadata. Installation CI tests
the resulting packages on Linux, Windows, Apple Silicon, and Intel Macs.

Install the build tool in Conda's base environment, where the `conda` command
discovers its subcommands. Builds and tests run in separate environments created
by conda-build. From the repository root:

```bash
conda install -n base --override-channels -c conda-forge conda-build
conda build conda/recipe --override-channels -c conda-forge
conda create -n gbnet-native --use-local --override-channels --strict-channel-priority -c conda-forge python=3.11 gbnet
conda run -n gbnet-native python -I scripts/smoke_install.py
```

The default build targets Python 3.11. Build another compatible Python variant
with `conda build conda/recipe --python 3.12 --override-channels -c conda-forge`.
The public feedstock should build the supported Python variants for each platform.

The installation workflow also produces a `conda-package-<runner>` artifact containing an
indexed local channel. Download and unpack it, then use its directory as a channel
(`conda create -n gbnet-native --override-channels -c ./conda-dist -c conda-forge
python=3.11 gbnet`). A local build or CI artifact does **not** make GBNet available
on a public Conda channel.

To make `conda install -c conda-forge gbnet` available to everyone:

1. Release the installation fixes to PyPI with a new version number.
2. Submit a recipe to [conda-forge staged-recipes](https://github.com/conda-forge/staged-recipes).
   Replace this recipe's local `source.path` with the released source archive URL
   and its SHA-256. Replace the `load_file_data` expression with the released
   version, and add the required `extra.recipe-maintainers` list. Preserve all
   runtime dependencies and the fitting smoke test.
3. Once accepted, maintain the resulting feedstock and let its builds publish the
   packages. Keep `pyproject.toml` and the Conda dependency
   bounds aligned when upgrading the native stack.

An owner-controlled Anaconda channel is another option: upload the tested package
there and document the actual channel name. This repository does not assume access
to an Anaconda publishing account or silently publish to one.
