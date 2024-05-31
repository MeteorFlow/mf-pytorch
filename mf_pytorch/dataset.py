import dataclasses
import functools
import io

import boto3
import boto3.s3
import torch
import numpy
from torch.utils.data import Dataset


@dataclasses.dataclass
class DateTimeRage:
    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None

    def get_file_prefix(self) -> str:
        prefix = ""

        if not self.year:
            return prefix
        prefix += str(self.year)

        if not self.month:
            return prefix
        prefix += str(self.month)

        if not self.day:
            return prefix
        prefix += str(self.day)

        if not self.hour:
            return prefix
        prefix += "T" + str(self.hour)

        if not self.minute:
            return prefix
        prefix += str(self.hour)

        return prefix


class MeteorFlowDataset(Dataset):
    def __init__(
        self,
        s3_endpoint_url: str,
        s3_access_key: str,
        s3_secret_key: str,
        region: str,
        date_time_range: DateTimeRage,
        features: list[str],
    ):
        self.region = region
        self.date_time_range = date_time_range
        self.features = features

        s3_client = boto3.resource(
            service_name="s3",
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            endpoint_url=s3_endpoint_url,
        )
        bucket_name = f"{self.region}-processed"
        self.s3_bucket = s3_client.Bucket(bucket_name)

    def __len__(self):
        return len(self._get_file_names())

    def __getitem__(self, idx) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        object_to_fetch = self._get_file_names()[idx]
        object = self.s3_bucket.Object(object_to_fetch.key)

        download_stream = io.BytesIO()
        object.download_fileobj(download_stream)

        download_stream.seek(0)
        data = numpy.load(download_stream)

        return torch.from_numpy(data)

    @functools.cache
    def _get_file_names(self) -> list[str]:
        object_prefix = self.date_time_range.get_file_prefix()
        objects_iter = self.s3_bucket.objects.filter(
            Prefix=f"{self.features[0]}/{object_prefix}"
        )

        return list(objects_iter)
