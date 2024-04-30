import polars as pl
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def get_content(row):
    try:
        r = requests.get(row, timeout=1)
        r.raise_for_status()
    except Exception:
        return ""
    return "" if r.status_code != 200 else BeautifulSoup(r.content, "lxml").text


def classification_training():
    classification = pl.read_csv(
        "./data/classification/classification.tsv",
        separator="\t",
        truncate_ragged_lines=True,
    )

    pbar = tqdm(total=len(classification), desc="Processing Rows:")
    classification = classification.with_columns(
        pl.col("URL")
        .map_elements(w_pbar(pbar, get_content), return_dtype=pl.String)
        .alias("content")
    )
    pbar.close()

    classification = classification.filter(pl.col("content") != "")
    classification.write_parquet("./data/classification/classification.parquet")
