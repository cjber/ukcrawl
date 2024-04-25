import os

from dagster import ConfigurableResource, EnvVar
from dagster_aws.s3 import S3Resource
from dask.distributed import Client, LocalCluster


class DaskResource(ConfigurableResource):
    n_workers: int

    def make_dask_cluster(self) -> Client:
        return Client(LocalCluster(n_workers=self.n_workers))


s3_resource = S3Resource(
    region_name="us-east-1",
    aws_access_key_id=EnvVar("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=EnvVar("AWS_SECRET_ACCESS_KEY"),
)
dask_resource = DaskResource(n_workers=min(os.cpu_count() or 1, 8))
