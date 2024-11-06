# Use this code if you want to create a new index

from elasticsearch import Elasticsearch, helpers, exceptions
import pandas as pd

# Load credentials from CSV
df = pd.read_csv('./elastic credential/watsonx_discovery_credential_tzwx24.csv')

# Set up Elasticsearch credentials
ELASTIC_USER = df.iloc[0]['username']
ELASTIC_PW = df.iloc[0]['password']
ELASTIC_HOST = df.iloc[0]['watsonx_discovery_url']  # WxD/elastic Endpoint
ELASTIC_PORT = df.iloc[0]['port']  # Port number
ELASTIC_CERT_FILE = "./elastic_certificate/es.cert"  # Path to cert file

# Define index and pipeline settings
INDEX_NAME_E5_EXTENDED = "index-e2-2" # define your index name
INGEST_PIPELINE_NAME = "e5-pipeline" # use the existing pipeline that you created in config-e5_embedding.ipynb, dont change or create a new pipeline 
MODEL_ID_E5 = ".multilingual-e5-small" # use the same model that you downloaded and deployed while config elastic

# Connect to Elasticsearch
client = Elasticsearch(
    f"{ELASTIC_HOST}:{ELASTIC_PORT}",
    basic_auth=(ELASTIC_USER, ELASTIC_PW),
    ca_certs=ELASTIC_CERT_FILE,
    verify_certs=True,
    request_timeout=120
)

# Delete the index if it already exists
client.indices.delete(index=INDEX_NAME_E5_EXTENDED, ignore=[400, 404])

# Create the new index with specified mappings
client.indices.create(
    index=INDEX_NAME_E5_EXTENDED,
    body={
        "mappings": {
            "_source": {
                "includes": [
                    "chunk_num",
                    "title",
                    "text",
                    "url",
                    "passage_embedding.predicted_value"
                ]
            },
            "properties": {
                "title": { "type": "keyword" },
                "text": { "type": "text" },
                "url": { "type": "keyword" },
                "chunk_num": { "type": "text" },
                "passage_embedding.predicted_value": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
)

print(f"Index '{INDEX_NAME_E5_EXTENDED}' created with custom mapping and existing pipeline '{INGEST_PIPELINE_NAME}'.")
