import gzip
from io import BytesIO

import boto3
import dask.bag as db
import duckdb
import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, asset
from retrying import retry
from warcio import ArchiveIterator

from src.common.utils import Consts, Paths, process_record, retry_policy
from src.partitions import archive_partition
from src.resources import DaskResource, S3Resource


@asset(
    retry_policy=retry_policy,
    compute_kind="s3",
    description="List of all available releases.",
)
def releases(s3: S3Resource) -> MaterializeResult:
    files = s3.get_client().list_objects_v2(
        Bucket=Consts.BUCKET,
        Prefix=Consts.PREFIX,
        Delimiter=Consts.DELIMITER,
    )

    all_releases = [file["Prefix"].split("/")[1] for file in files["CommonPrefixes"]]
    releases = [
        release
        for release in all_releases
        if release.startswith("CC-MAIN")
        and any(year in release for year in Consts.ARCHIVE_YEARS)
    ]

    complete_releases = []
    for release in releases:
        s3_objects = s3.get_client().list_objects_v2(
            Bucket=Consts.BUCKET,
            Prefix=f"{Consts.PREFIX}{release}/wet.paths.gz",
            Delimiter=Consts.DELIMITER,
        )
        if s3_objects.get("Contents"):
            complete_releases.append(release)

    with open(Paths.RELEASES, "w") as f:
        f.write("\n".join(reversed(complete_releases)))

    return MaterializeResult(metadata={"num_releases": len(complete_releases)})


@asset(
    partitions_def=archive_partition,
    deps=[releases],
    retry_policy=retry_policy,
    compute_kind="s3",
    description="List of WET files for a specific release.",
)
def wet_files_list(context: AssetExecutionContext, s3: S3Resource) -> MaterializeResult:
    release = context.partition_key

    if (Paths.DONE / f"{release}.done").exists():
        context.log.info(f"Skipping {release}")
        return MaterializeResult(metadata={"num_files": "Unknown"})

    out_file_path = Paths.WET_LIST / f"{release}.txt"
    if not out_file_path.exists():
        out_file_path.touch()

    s3_objects = s3.get_client().list_objects_v2(
        Bucket=Consts.BUCKET,
        Prefix=f"{Consts.PREFIX}{release}/wet.paths.gz",
        Delimiter=Consts.DELIMITER,
    )

    if not s3_objects.get("Contents"):
        raise ValueError(f"No WET files found for {release}")

    for obj in s3_objects.get("Contents", []):
        key = obj["Key"]
        response = s3.get_client().get_object(Bucket=Consts.BUCKET, Key=key)
        compressed_data = response["Body"].read()
        with gzip.GzipFile(fileobj=BytesIO(compressed_data), mode="rb") as f:
            files = f.read().strip().decode().split("\n")

    with open(out_file_path, "w") as f:
        f.write("\n".join(files))
    return MaterializeResult(metadata={"num_files": len(files)})


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)
def _extract_wet(wet_url):
    name = wet_url.split("/")[-1].split(".")[0]
    release = wet_url.split("/")[1]
    s3 = boto3.client("s3")

    out_file = Paths.DATA / f"{release}" / f"{name}.parquet"
    if out_file.exists():
        return

    response = s3.get_object(Bucket=Consts.BUCKET, Key=wet_url)

    dicts = []
    for record in ArchiveIterator(BytesIO(response["Body"].read())):
        if record.rec_type == "conversion":
            url = record.rec_headers.get_header("WARC-Target-URI")
            if Consts.UK_URL not in url:
                continue
            processed = process_record(record)
            dicts.append(processed)
    pd.DataFrame(dicts).to_parquet(out_file)


@asset(
    partitions_def=archive_partition,
    deps=[wet_files_list],
    compute_kind="dask",
    description="UK Specific URLs for each WET file in a release, saved as Parquet files.",
)
def uk_files(context: AssetExecutionContext, local_dask: DaskResource) -> None:
    release = context.partition_key
    wet_list = Paths.WET_LIST / f"{release}.txt"

    parquets_dir = Paths.DATA / release
    if not parquets_dir.exists():
        parquets_dir.mkdir()

    with open(wet_list) as f:
        files = f.read().split("\n")

    client = local_dask.make_dask_cluster()

    with client:
        files_scattered = client.scatter(files)
        b = db.from_sequence(files_scattered, npartitions=100)  # type: ignore
        b.map(_extract_wet).compute()


@asset(
    partitions_def=archive_partition,
    deps=[uk_files],
    compute_kind="duckdb",
    description="Combined UK files into a single parquet.",
)
def combined_files(context: AssetExecutionContext) -> MaterializeResult:
    release = context.partition_key
    parquets_dir = Paths.DATA / release

    duckdb.sql(
        f"""
        COPY 
            (SELECT * FROM parquet_scan('{str(parquets_dir)}/*.parquet'))
        TO '{str(Paths.ARCHIVE / release)}.parquet' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )
    release_done = Paths.DONE / f"{release}.done"
    release_done.touch()

    for child in parquets_dir.iterdir():
        child.unlink()
    parquets_dir.rmdir()

    out_file = Paths.ARCHIVE / f"{release}.parquet"
    if out_file.exists():
        num_rows = duckdb.sql(
            f"SELECT COUNT(*) FROM parquet_scan('{out_file}')"
        ).fetchone() or ["Unknown"]
        return MaterializeResult(metadata={"num_rows": num_rows[0]})
    return MaterializeResult(metadata={"num_rows": "Unknown"})
