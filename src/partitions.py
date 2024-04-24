from dagster import DynamicPartitionsDefinition

archive_partition = DynamicPartitionsDefinition(name="archive")
