FROM python:3.12

ENV VIRTUAL_ENV=/usr/local
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

ENV DAGSTER_HOME=/opt/dagster/dagster_home/
RUN mkdir -p $DAGSTER_HOME /opt/dagster/app
COPY dagster.yaml $DAGSTER_HOME

WORKDIR /opt/dagster/app

COPY workspace.yaml requirements.txt pyproject.toml .env ./
RUN /root/.cargo/bin/uv pip install --no-cache -r requirements.txt
