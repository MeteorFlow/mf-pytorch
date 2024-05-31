import os

from mf_pytorch.dataset import MeteorFlowDataset, DateTimeRage


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


if __name__ == "__main__":
    main()
