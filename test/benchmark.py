import os
from multiprocessing import Pool

import pendulum

from mf_pytorch.dataset import MeteorFlowDataset


def fetch_data(process_id):
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_ACCESS_SECRET = os.getenv("S3_ACCESS_SECRET")

    assert S3_ENDPOINT_URL is not None
    assert S3_ACCESS_KEY is not None
    assert S3_ACCESS_SECRET is not None

    mf_dataset = MeteorFlowDataset(
        S3_ENDPOINT_URL,
        S3_ACCESS_KEY,
        S3_ACCESS_SECRET,
        region="nha-be",
        features=["reflectivity"],
        start_date_time=pendulum.datetime(2019, 5, 12, 0, 0),
        end_date_time=pendulum.datetime(2019, 5, 12, 23, 19),
    )

    for dataset in mf_dataset:
        print(f"{pendulum.now()} - Process {process_id} fetching data")
        pass


def main():
    with Pool(processes=5) as pool:
        pool.map(fetch_data, range(5))


if __name__ == "__main__":
    main()
