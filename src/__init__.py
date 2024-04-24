from dagster import Definitions, load_assets_from_modules

from src.assets import process, retrieve
from src.jobs import extract_uk_urls_job, process_job, releases_job
from src.resources import dask_resource, s3_resource
from src.schedules import release_schedule
from src.sensors import process_sensor, release_sensor

retrieve_assets = load_assets_from_modules(modules=[retrieve], group_name="retrieve")
process_assets = load_assets_from_modules(modules=[process], group_name="process")

defs = Definitions(
    assets=[*retrieve_assets, *process_assets],
    jobs=[releases_job, extract_uk_urls_job, process_job],
    resources={"local_dask": dask_resource, "s3": s3_resource},
    sensors=[release_sensor, process_sensor],
    schedules=[release_schedule],
)
