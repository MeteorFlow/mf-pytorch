import os

from mf_pytorch.dataset import MeteorFlowDataset, DateTimeRage
import pendulum


def main():
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_ACCESS_SECRET = os.getenv("S3_ACCESS_SECRET")

    assert S3_ENDPOINT_URL is not None
    assert S3_ACCESS_KEY is not None
    assert S3_ACCESS_SECRET is not None

    dataset = MeteorFlowDataset(
        S3_ENDPOINT_URL,
        S3_ACCESS_KEY,
        S3_ACCESS_SECRET,
        region="nha-be",
        features=["reflectivity"],
        date_time_range=DateTimeRage(year=2019),
    )

    print(len(dataset))
    print(dataset[0])

    dataset2 = MeteorFlowDataset(
        S3_ENDPOINT_URL,
        S3_ACCESS_KEY,
        S3_ACCESS_SECRET,
        region="nha-be",
        features=["reflectivity"],
        start_date_time=pendulum.datetime(2019, 5, 10, 23, 10),
        end_date_time=pendulum.datetime(2019, 5, 10, 23, 19),
    )

    print(len(dataset2))
    print(dataset2[0])


if __name__ == "__main__":
    main()
