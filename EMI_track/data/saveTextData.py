import pandas as pd
import os
import time

csv_file = "./text/audio2text.csv"
output_dir = "./text/"
os.makedirs(output_dir, exist_ok=True)

column_names = ["filename", "text"]
df = pd.read_csv(csv_file, encoding="ISO-8859-1", header=None, skiprows=1, names=column_names, index_col=False)

for _, row in df.iterrows():
    filename = row["filename"].strip().replace(".mp3", ".txt")
    text_content = str(row["text"]).strip()

    # print(filename)
    # print(text_content)
    # time.sleep(1)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

print(f"文件已成功存入 {output_dir} 目录")
