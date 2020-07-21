import json
import os
from reface import env


bucket = "data-for-refaceai-test"


def get_s3_client():
    if not hasattr(get_s3_client, "s3client"):
        import boto3

        creds = json.load(open(os.path.join(env.repo_root, "reface/aws-creds.json")))
        s3client = boto3.client(
            "s3",
            aws_access_key_id=creds["access_token"],
            aws_secret_access_key=creds["secret_token"],
            region_name=creds["region"],
        )
        get_s3_client.s3client = s3client

    return get_s3_client.s3client
