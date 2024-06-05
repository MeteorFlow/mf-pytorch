import os

from mf_pytorch.dataset import MeteorFlowDataset
import pendulum
import wradlib as wrl

import numpy as np


def main():
    cb = np.empty([2, 2])

    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_ACCESS_SECRET = os.getenv("S3_ACCESS_SECRET")

    assert S3_ENDPOINT_URL is not None
    assert S3_ACCESS_KEY is not None
    assert S3_ACCESS_SECRET is not None

    dataset2 = MeteorFlowDataset(
        S3_ENDPOINT_URL,
        S3_ACCESS_KEY,
        S3_ACCESS_SECRET,
        region="nha-be",
        features=["reflectivity"],
        start_date_time=pendulum.datetime(2019, 5, 10, 23, 10),
        end_date_time=pendulum.datetime(2019, 5, 10, 23, 19),
    )

    R_past = []
    for reflectivity_data in dataset2:
        cmax_data = np.array(np.max(reflectivity_data, axis=0))
        radar_Z = wrl.trafo.idecibel(cmax_data)
        R = wrl.zr.z_to_r(radar_Z, a=200.0, b=1.6)
        reflectivity_data[:] = R
        R = reflectivity_data[0]

        extra_left, extra_right = 128, 128
        extra_top, extra_bottom = 32, 32

        R = np.pad(
            R,
            ((extra_top, extra_bottom), (extra_left, extra_right)),
            mode="constant",
            constant_values=0,
        )
        R = R[cb[0][0] : cb[0][1], cb[1][0] : cb[1][1]]
        R_past.append(R)

    R_past = np.stack(R_past, axis=0)


if __name__ == "__main__":
    main()
