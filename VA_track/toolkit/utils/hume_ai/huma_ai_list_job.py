import asyncio
import os
from hume.client import HumeClient

api_key="LWKcin9431YI9ZdgEH3li7z63faVjYeIZqlSKa1hKfOrolQS"

client = HumeClient(
    api_key=api_key, # Defaults to HUME_API_KEY
)
response=client.expression_measurement.batch.list_jobs()
print(response)
print("done")