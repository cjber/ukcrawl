from dagster import define_asset_job

from src.assets.process import class_file, ner_file, postcodes_file
from src.assets.retrieve import (
    combined_files,
    releases,
    uk_files,
    wet_files_list,
)

releases_job = define_asset_job("releases_job", selection=[releases])
extract_uk_urls_job = define_asset_job(
    "extract_uk_urls_job",
    selection=[wet_files_list, uk_files, combined_files],
)
process_job = define_asset_job(
    "process_job", selection=[postcodes_file, ner_file, class_file]
)
