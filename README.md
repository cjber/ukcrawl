<div align="center">

# UKCrawl

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cjber/ukcrawl/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/cjber/ukcrawl.svg)](https://github.com/cjber/ukcrawl/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/cjber/ukcrawl.svg)](https://github.com/cjber/ukcrawl/pulls)

UKCrawl is a Python library for retrieving and processing data from UK websites using the [Common Crawl](https://commoncrawl.org/) archives. It queries [AWS S3](https://aws.amazon.com/s3) for data retrieval, then processes using [Named Entity Recognition](https://huggingface.co/docs/transformers/tasks/token_classification) for extracting named entities, [DuckDB](https://duckdb.org) for extracting postcodes, and a [HuggingFace transformers model](TBD) to classify webpages.

![Asset Linkage](https://github.com/cjber/ukcrawl/assets/44099524/152730d1-77e3-4daa-803b-c82fde390428)

</div>

## Initiate Dagster Orchestration

This project uses Dagster to monitor daily for new Common Crawl archives. Within `src/common/utils.py` there is a list of years that are retrieved. For any processed files that are missing, Dagster initiates a job. To run this orchestration pipeline:

1. **Install Podman and Podman Compose**: Ensure that Podman and Podman Compose are installed on your system. You can install them using your package manager or by following the instructions provided in their respective documentation.

2. **Clone the Repository**: Clone the UKCrawl repository from GitHub using the following command:

    ```bash
    git clone https://github.com/cjber/ukcrawl.git
    ```

3. **Navigate to the Project Directory**: Change your current directory to the root directory of the UKCrawl project:

    ```bash
    cd ukcrawl
    ```
5. **Start the Containers**: Use Podman Compose to start the containers defined in the `compose.yml` file:

    ```bash
    podman-compose up -d
    ```

6. **Access Dagster Web Interface**: Once the containers are up and running, access the Dagster web interface by navigating to `http://localhost:3000` in your web browser.
