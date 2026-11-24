# Third-Party Notices

## torchhydro

- Upstream repository: <https://github.com/OuyangWenyu/torchhydro>
- License: BSD 3-Clause

This repository includes code adapted from upstream `torchhydro` components. The water-balance models and training loss are adapted from the authors' `torchhydro/models/dplgr4j_dy_waterbal.py` and `torchhydro/models/crits.py`. The original experiment entry point is preserved in `examples/dpl_gr4j_waterbal.py`. The upstream BSD 3-Clause license text is included in `LICENSES/torchhydro-BSD-3-Clause.txt`.

## Data Notice

The original Lianghe experiment requires `SXdata`, which is not bundled. The standalone water-balance example uses the CAMELS CSV included under `data/`.
