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

