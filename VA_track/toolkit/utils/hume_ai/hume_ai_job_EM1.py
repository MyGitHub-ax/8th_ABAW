import asyncio
import os
from hume.client import HumeClient
from hume.expression_measurement.batch.types.inference_base_request import InferenceBaseRequest
import json

api_key="LWKcin9431YI9ZdgEH3li7z63faVjYeIZqlSKa1hKfOrolQS"

client = HumeClient(
    api_key=api_key, # Defaults to HUME_API_KEY
)

# 文件的路径
file_path = os.path.join("/data/wenzhuofan/Data/AVEC2013/RAW/train","203_1_03.mp4")
# 打开并读取文件的二进制数据
with open(file_path, "rb") as f:
    file_content = f.read()
# 构建 File 对象
file = (file_path.split("/")[-1], file_content, "video/mp4")

inference_request = InferenceBaseRequest(
    models={"face": {}},  # 假设 models 是一个字典，具体内容根据需求填写
    # notify=True  # 请求完成后发送通知
)
response_id=client.expression_measurement.batch.start_inference_job_from_local_file(file=file,json=inference_request)
print(f'response_id:{response_id}')
print("done")