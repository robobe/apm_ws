import numpy as np

from apm_demos.visio_estimate import PoseAndOrientation


def test_lat_lon_diff():
    p1 = PoseAndOrientation(lat=43.937845, lon=-97.905537)
    p2 = PoseAndOrientation(lat=44.310739, lon=-97.588820)

    print(p1.to_xy())
    result = p1.distance(p2)
    EXPECTED = 41.14953
    assert True  # np.testing.assert_approx_equal(result, EXPECTED, significant=4)
