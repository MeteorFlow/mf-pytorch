import dataclasses
import functools
import io
from typing import Iterable

import boto3
import boto3.s3
import numpy
import pendulum
import torch
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
        features: list[str],
        date_time_range: DateTimeRage | None = None,
        start_date_time: pendulum.DateTime | None = None,
        end_date_time: pendulum.DateTime | None = None,
    ):
        self.region = region
        self.features = features
        self.date_time_range = date_time_range
        self.start_date_time = start_date_time
        self.end_date_time = end_date_time

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
        object = self.s3_bucket.Object(object_to_fetch)

        download_stream = io.BytesIO()
        object.download_fileobj(download_stream)

        download_stream.seek(0)
        data = numpy.load(download_stream)

        return torch.from_numpy(data)

    @functools.cache
    def _get_file_names(self) -> list[str]:
        if self.date_time_range:
            object_prefix = self.date_time_range.get_file_prefix()
            objects_iter = self.s3_bucket.objects.filter(
                Prefix=f"{self.features[0]}/{object_prefix}"
            )

            return list(map(lambda object_summary: object_summary.key, objects_iter))
        else:
            all_keys = self._get_all_keys()

            if self.start_date_time:
                start_date_format = self.start_date_time.format("[reflectivity]/YYYYMMDDTHHmmssZ")
                valid_keys = filter(lambda keys: keys >= start_date_format, all_keys)

            if self.end_date_time:
                end_date_format = self.end_date_time.format("[reflectivity]/YYYYMMDDTHHmmssZ")
                valid_keys = filter(lambda keys: keys <= end_date_format, valid_keys)

            return list(valid_keys)

    @functools.cache
    def _get_all_keys(self) -> Iterable[str]:
        return map(
            lambda object_summary: object_summary.key,
            self.s3_bucket.objects.filter(Prefix=f"{self.features[0]}/"),
        )
