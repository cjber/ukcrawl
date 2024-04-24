from pathlib import Path

import datasets
import duckdb
import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from src.common.utils import Consts, Paths, extract_domain
from src.partitions import archive_partition

datasets.disable_caching()


def _create_dataset(in_file: Path) -> datasets.Dataset:
    wet_dataset = (
        load_dataset(
            "parquet",
            data_files=str(in_file),
            streaming=False,
            split="train",
        )
        .rename_column("content", "text")
        .filter(lambda example: extract_domain(example["url"], Consts.UK_URL))
    )
    return wet_dataset  # type: ignore


@asset(
    partitions_def=archive_partition,
    deps=[AssetKey("combined_files")],
    description="NER Parquet file for a specific release.",
    compute_kind="transformers",
)
def ner_file(context: AssetExecutionContext) -> MaterializeResult:
    release = context.partition_key
    in_file = Paths.ARCHIVE / f"{release}.parquet"
    out_file = Paths.NER / f"{release}_ner.parquet"
    if out_file.exists():
        return MaterializeResult(metadata={"num_records": 0, "preview": "No data"})

    wet_dataset = _create_dataset(in_file)
    if not wet_dataset:
        pl.DataFrame().write_parquet(out_file)
        return MaterializeResult(metadata={"num_records": 0, "preview": "No data"})

    model = "dbmdz/bert-large-cased-finetuned-conll03-english"
    batch_size = 8
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=model,
        device="cuda",
        batch_size=batch_size,
        aggregation_strategy="first",
    )

    outs = []
    for idx, example in enumerate(ner_pipeline(KeyDataset(wet_dataset, "text"))):
        url: dict = wet_dataset[idx]
        del url["text"]
        outs.append({"idx": idx, "ner": example} | url)

    df = pl.DataFrame(outs).filter(pl.col("ner").list.len() > 0)
    df.write_parquet(out_file)

    return MaterializeResult(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(
                df.head().explode("ner").to_pandas().to_markdown() or "No data"
            ),
        }
    )


@asset(
    partitions_def=archive_partition,
    deps=[AssetKey("combined_files")],
    description="Postcodes Parquet file for a specific release.",
    compute_kind="duckdb",
)
def postcodes_file(context: AssetExecutionContext) -> MaterializeResult:
    release = context.partition_key
    in_file = Paths.ARCHIVE / f"{release}.parquet"
    out_file = Paths.PC / f"{release}_pc.parquet"
    if out_file.exists():
        return MaterializeResult(metadata={"num_records": 0, "preview": "No data"})

    duckdb.sql(
        f"""
        COPY
            (
            SELECT 
                url,
                length,
                lang,
                date,
                record_id,
                refers_to,
                regexp_extract_all(content, '{Consts.PCRE}') AS postcodes
            FROM read_parquet('{in_file}')
            WHERE len(postcodes) > 0
            )
        TO '{out_file}' (FORMAT 'parquet', COMPRESSION 'zstd')
        """
    )

    df = pl.read_parquet(out_file)
    return MaterializeResult(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(
                df.head().to_pandas().to_markdown() or "No data"
            ),
        }
    )


@asset(
    partitions_def=archive_partition,
    deps=[AssetKey("combined_files")],
    description="Classification Parquet file for a specific release.",
    compute_kind="transformers",
)
def class_file(context: AssetExecutionContext) -> MaterializeResult:
    release = context.partition_key
    in_file = Paths.ARCHIVE / f"{release}.parquet"
    out_file = Paths.CLASS / f"{release}_class.parquet"
    if out_file.exists():
        return MaterializeResult(metadata={"num_records": 0, "preview": "No data"})

    wet_dataset = _create_dataset(in_file)
    if not wet_dataset:
        pl.DataFrame().write_parquet(out_file)
        return MaterializeResult(metadata={"num_records": 0, "preview": "No data"})

    model = "alimazhar-110/website_classification"
    batch_size = 8
    ner_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        device="cuda",
        batch_size=batch_size,
        truncation=True,
    )

    outs = []
    for idx, example in enumerate(ner_pipeline(KeyDataset(wet_dataset, "text"))):
        url: dict = wet_dataset[idx]
        del url["text"]
        outs.append({"idx": idx} | url | example)

    df = pl.DataFrame(outs).rename({"label": "classification", "score": "confidence"})
    df.write_parquet(out_file)

    return MaterializeResult(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(
                df.head().to_pandas().to_markdown() or "No data"
            ),
        }
    )
