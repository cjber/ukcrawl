from dagster import (
    AssetKey,
    DagsterRunStatus,
    DefaultSensorStatus,
    RunRequest,
    RunsFilter,
    SensorEvaluationContext,
    SkipReason,
    asset_sensor,
    sensor,
)

from src.common.utils import Paths
from src.jobs import extract_uk_urls_job, process_job
from src.partitions import archive_partition


@sensor(job=extract_uk_urls_job, default_status=DefaultSensorStatus.RUNNING)
def release_sensor(context: SensorEvaluationContext):
    if not Paths.RELEASES.exists():
        yield SkipReason(f"{Paths.RELEASES} does not exist.")
        return

    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name=extract_uk_urls_job.name,
            statuses=[DagsterRunStatus.QUEUED, DagsterRunStatus.STARTED],
        ),
    )
    if runs:
        yield SkipReason(f"{extract_uk_urls_job.name} is already running or queued.")
        return

    with open(Paths.RELEASES) as f:
        releases = f.read().splitlines()

    context.instance.add_dynamic_partitions(
        archive_partition.name or "archive", releases
    )
    remaining_releases = [
        release for release in releases if not (Paths.DONE / f"{release}.done").exists()
    ]
    if remaining_releases:
        yield RunRequest(partition_key=remaining_releases[0])


@asset_sensor(
    asset_key=AssetKey("combined_files"),
    job=process_job,
    default_status=DefaultSensorStatus.RUNNING,
)
def process_sensor(context: SensorEvaluationContext):
    if not Paths.RELEASES.exists():
        yield SkipReason(f"{Paths.RELEASES} does not exist.")
        return

    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name=process_job.name,
            statuses=[DagsterRunStatus.QUEUED, DagsterRunStatus.STARTED],
        ),
    )
    if runs:
        yield SkipReason(f"{process_job.name} is already running or queued.")
        return

    with open(Paths.RELEASES) as f:
        releases = f.read().splitlines()

    context.instance.add_dynamic_partitions(
        archive_partition.name or "archive", releases
    )
    remaining_releases = [
        release
        for release in releases
        if (not (Paths.NER / f"{release}_ner.parquet").exists())
        or (not (Paths.PC / f"{release}_pc.parquet").exists())
        or (not (Paths.CLASS / f"{release}_class.parquet").exists())
    ]
    if remaining_releases:
        yield RunRequest(partition_key=remaining_releases[0])
