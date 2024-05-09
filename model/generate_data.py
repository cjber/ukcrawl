import backoff
import openai
import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.common.utils import Consts, extract_domain

load_dotenv()
client = OpenAI()

datagen_model = "gpt-3.5-turbo"


def load_data():
    content = (
        pl.scan_parquet("./data/out/archive/*.parquet")
        .select(["url", "content"])
        .filter(
            pl.col("url").map_elements(
                lambda x: extract_domain(x, subset=Consts.UK_URL),
                return_dtype=pl.Boolean,
            )
        )
        .collect()
    )
    with open("./data/categories.txt") as f:
        categories = f.read().splitlines()
    return content, categories


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def generate_data(content, categories):
    outputs = []
    for row in tqdm(content.rows(named=True)):
        question = f"""
        Classify the content of the following URL into a single category from the provided list.
        RETURN ONLY THE CATEGORY NAME, DO NOT WRITE ANY OTHER TEXT.

        URL: {row['url']}

        Content: {row['content'][:5000] + '...' if len(row['content']) > 5000 else row['content']}

        Categories: {categories}
        """
        response = completion_with_backoff(
            model=datagen_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to generate annotated data.",
                },
                {"role": "user", "content": question},
            ],
        )
        res = response.choices[0].message.content
        outputs.append({"url": row["url"], "category": res})

    pl.DataFrame(outputs).join(content, on="url").write_parquet(
        "./data/synthetic_data.parquet"
    )


if __name__ == "__main__":
    content, categories = load_data()
    generate_data(content, categories)
