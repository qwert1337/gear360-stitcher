# gear360-stitcher

Stitch photos and videos from the **Samsung Gear 360 (SM-C200)** dual-fisheye
camera into equirectangular panoramas, using per-lens Double Sphere
calibrations (plus vignette models) and a front-to-back extrinsic pose
(`calibration.json`, `vignette_*.json`, `R_t.txt`).

The intrinsics (lens calibrations) can be produced with
[online-camera-calibration.com](https://online-camera-calibration.com/). That
tool does not yet provide the extrinsics (`R_t.txt`); a tool for them is
planned. Until then, the calibration bundled with this package is used by
default.

## Install

The easiest way is to install it as an isolated command-line tool, so that no
virtual environment has to be managed and the system Python stays untouched.

From PyPI:

    pipx install gear360-stitcher      # or: uv tool install gear360-stitcher

From the GitHub repository (latest development version):

    pipx install git+https://github.com/qwert1337/gear360-stitcher
    # or: uv tool install git+https://github.com/qwert1337/gear360-stitcher

From a local checkout:

    git clone https://github.com/qwert1337/gear360-stitcher
    cd gear360-stitcher
    pipx install .                     # or: uv tool install .

To try it once without installing: `uvx --from gear360-stitcher gear360-stitch --help`.

With plain `pip` (inside a virtual environment): `pip install gear360-stitcher`,
or `pip install -e ".[test]"` from a checkout for development.

Video output additionally needs `ffmpeg` on `PATH`.

## Usage

    gear360-stitch 360_0439.JPG -o out.jpg
    gear360-stitch 360_0439.MP4 -o out.mp4

By default the calibration bundled in the package is used. To use your own:

    gear360-stitch in.JPG -o out.jpg \
        --calib-front FRONT_DIR --calib-back BACK_DIR --extrinsics R_t.txt

`gear360-stitch --help` lists all quality/speed options.

## Calibration format

Each lens directory holds `calibration.json` and the vignette files
(`vignette_circle.json`, `vignette_polynomial.json`):

```json
{"resolution": [3872, 3872],
 "params_ds": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "alpha": 0, "Xi": 0}}
```
