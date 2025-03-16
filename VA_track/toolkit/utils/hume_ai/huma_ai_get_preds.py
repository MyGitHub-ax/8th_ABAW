import asyncio
import os
import json
from hume.client import HumeClient

api_key="LWKcin9431YI9ZdgEH3li7z63faVjYeIZqlSKa1hKfOrolQS"

client = HumeClient(
    api_key=api_key, # Defaults to HUME_API_KEY
)
response_id='c56ec0c9-778d-482c-a468-edf8b0c7ecba'
pred=client.expression_measurement.batch.get_job_predictions(id=response_id)
print(pred)
print("done")