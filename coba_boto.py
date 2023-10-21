import boto3
import joblib

# Vultr Object Storage configuration
endpoint_url = "https://sgp1.vultrobjects.com"
access_key = "43KGZL0JXCW1N1M1X2LS"
secret_key = "uPzHK2eWp6zcBDLgmfCaIWu4pPRo6k7v5IMDr2Gh"
bucket_name = "lilz"
model_key = "coba.pkl"  # Adjust the key as needed

# Initialize S3 client
s3 = boto3.client('s3', endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# Download the model
with open("test.pkl", "wb") as file:
    s3.download_fileobj(bucket_name, model_key, file)

# Load the model
model = joblib.load("test.pkl")