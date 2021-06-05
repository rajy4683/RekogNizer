import boto3
import os
import io
import json
import base64
#from requests_toolbelt.multipart import decoder
from botocore.config import Config
import torch

#os.environ['AWS_CONFIG_FILE']="/content/awscliini.txt"
### 

def set_aws_params_for_boto3(access_key, access_secret, default_region, default_bucket):
    os.environ['AWS_ACCESS_KEY_ID']=access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = access_secret
    os.environ['AWS_DEFAULT_REGION'] = default_region
    os.environ['S3_DEFAULT_BUCKET']=default_bucket


def model_load_from_s3(bucket_name, obj_name):
    try:
        s3 = boto3.client('s3', 
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        obj = s3.get_object(Bucket=bucket_name, Key=obj_name)
        print( "Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read() )
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print( "Model Loaded")
        return model
    except Exception as e:
        print("Base Model Loading failed!",repr(e))
        raise(e)

def upload_model_to_s3(bucket_name, file_name, obj_name):
    s3 = boto3.client('s3', 
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    with open(file_name, "rb") as f:
        s3.upload_fileobj(f, bucket_name, obj_name)
