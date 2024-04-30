import re
import warnings
from pathlib import Path

import tldextract
from dagster import Backoff, ExperimentalWarning, RetryPolicy
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=ExperimentalWarning)

load_dotenv()


class Consts:
    BUCKET = "commoncrawl"
    PREFIX = "crawl-data/"
    DELIMITER = "/"

    UK_URL = ".uk/"
    ARCHIVE_YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]

    PCRE = r"[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}"

    CLASS_MODEL = "distilbert/distilbert-base-uncased"


class Labels:
    ID2LABEL = {
        0: "Government, Law & Politics",
        1: "Libraries, Archives and Museums",
        2: "Publishing, Printing and Bookselling",
        3: "Popular Science",
        4: "Social Problems and Welfare",
        5: "Crime, Criminology, Police and Prisons",
        6: "Life Sciences",
        7: "Politics, Political Theory and Political Systems",
        8: "Sports and Recreation",
        9: "Literature",
        10: "Company Web Sites",
        11: "Law and Legal System",
        12: "Digital Society",
        13: "History",
        14: "Environment",
        15: "Business, Economy & Industry",
        16: "Science & Technology",
        17: "Society & Culture",
        18: "Education & Research",
        19: "Travel & Tourism",
        20: "Arts & Humanities",
        21: "Religion",
        22: "Medicine & Health",
        23: "Computer Science, Information Technology and Web Technology",
    }
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}
    COUNT = len(ID2LABEL)


class Paths:
    DATA = Path("data")
    RELEASES = DATA / "releases.txt"
    WET_LIST = DATA / "wet_files"

    OUTPUT = DATA / "out"
    ARCHIVE = OUTPUT / "archive"
    DONE = OUTPUT / "done"

    PROCESSED = DATA / "processed"
    PC = PROCESSED / "pc"
    NER = PROCESSED / "ner"
    CLASS = PROCESSED / "class"

    @classmethod
    def ensure_directories_exist(cls):
        cls.DONE.mkdir(parents=True, exist_ok=True)
        cls.ARCHIVE.mkdir(parents=True, exist_ok=True)
        cls.WET_LIST.mkdir(parents=True, exist_ok=True)
        cls.PC.mkdir(parents=True, exist_ok=True)
        cls.NER.mkdir(parents=True, exist_ok=True)
        cls.CLASS.mkdir(parents=True, exist_ok=True)


Paths.ensure_directories_exist()


def extract_domain(url: str, subset: str) -> bool:
    parsed_url = tldextract.extract(url)
    is_single_http = len(re.findall("http|https", url)) <= 1
    is_uk_domain = url.endswith(subset)
    is_no_subdomain_or_www = parsed_url.subdomain in ["", "www"]

    return is_no_subdomain_or_www and is_uk_domain and is_single_http


def process_record(record) -> dict:
    return {
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "content": record.content_stream().read().decode("utf-8", errors="ignore"),
        "length": record.rec_headers.get_header("Content-Length"),
        "lang": record.rec_headers.get_header("WARC-Identified-Content-Language"),
        "date": record.rec_headers.get_header("WARC-Date"),
        "record_id": record.rec_headers.get_header("WARC-Record-ID"),
        "refers_to": record.rec_headers.get_header("WARC-Refers-To"),
    }


retry_policy = RetryPolicy(max_retries=5, delay=60, backoff=Backoff.EXPONENTIAL)
