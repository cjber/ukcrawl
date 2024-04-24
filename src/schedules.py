from dagster import DefaultScheduleStatus, ScheduleDefinition

from src.jobs import releases_job

release_schedule = ScheduleDefinition(
    job=releases_job,
    cron_schedule="0 0 * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)
