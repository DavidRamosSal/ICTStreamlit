import os

from google.cloud import storage
from google.oauth2 import service_account

import streamlit as st


def download_folder(bucket_name, prefix, local_folder, credentials):
    client = storage.Client(credentials=credentials)

    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        relative_path = blob.name[len(prefix) :]

        # If relative_path is empty or represents a directory, skip it
        if not relative_path or relative_path.endswith("/"):
            continue

        # Create local destination path
        local_path = os.path.join(local_folder, relative_path)

        # Create directories if necessary
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def download_file(bucket_name, blob_name, local_path, credentials):
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_name} to {local_path}")


def check_and_download(bucket_name, blob_name, local_path, credentials):
    if not os.path.exists(local_path):
        download_file(bucket_name, blob_name, local_path, credentials)


bucket_name = "data-cnc-predictive-maintenance"

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcs_connections"]
)
