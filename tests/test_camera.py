import json
import numpy as np

from gear360_stitcher.camera import DoubleSphere, ParamsDoubleSphere, load_calibration

PARAMS = ParamsDoubleSphere(500.0, 500.0, 960.0, 960.0, 0.6, -0.1)


def test_center_ray_projects_to_principal_point():
    uv, valid = DoubleSphere(PARAMS).world2cam(np.array([[[0.0, 0.0, 1.0]]]))
    np.testing.assert_allclose(uv[0, 0], [960.0, 960.0], atol=1e-4)
    assert valid.all()


def test_ray_behind_camera_is_invalid():
    _, valid = DoubleSphere(PARAMS).world2cam(np.array([[[0.0, 0.0, -1.0]]]))
    assert not valid.any()


def test_load_calibration_json(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"resolution": [1920, 1920], "params_ds": PARAMS._asdict()}))
    params, res = load_calibration(path)
    assert params == PARAMS and res == (1920, 1920)



def test_resolve_calib_dir_caches_only_provided_files(tmp_path, monkeypatch):
    from gear360_stitcher import cli

    monkeypatch.setattr(cli, "CACHE_DIR", tmp_path / "cache")
    user = tmp_path / "user"
    user.mkdir()
    (user / "calibration.json").write_text("{}")

    assert cli.resolve_calib_dir(None, "front") == cli.DATA_DIR / "front"
    assert not (tmp_path / "cache").exists()

    out = cli.resolve_calib_dir(user, "front")
    assert [p.name for p in (tmp_path / "cache" / "front").iterdir()] == ["calibration.json"]
    assert (out / "calibration.json").read_text() == "{}"
    for name in cli.CALIB_FILES[1:]:
        assert (out / name).read_bytes() == (cli.DATA_DIR / "front" / name).read_bytes()

    out = cli.resolve_calib_dir(None, "front")
    assert (out / "calibration.json").read_text() == "{}"
