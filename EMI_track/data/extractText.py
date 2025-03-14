import whisper
import os
import pandas as pd
from tqdm import tqdm

model = whisper.load_model("base")

# 转录音频文件
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']


def audio2text(input_folder, output_file):
    audio_files = [f for f in os.listdir(input_folder)]

    transcriptions = []

    for audio_file in tqdm(audio_files, desc="Transcribing Audio Files", unit="file"):
        file_path = os.path.join(input_folder, audio_file)
        try:
            text = transcribe_audio(file_path)
            # 将文件名和转录文本添加到 DataFrame
            transcriptions.append({"file_name": audio_file, "transcription": text})
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            transcriptions.append({"file_name": audio_file, "transcription": "Error during transcription"})

    transcriptions_df = pd.DataFrame(transcriptions)

    # 保存为CSV文件
    transcriptions_df.to_csv(output_file, index=False)
    print(f"Transcription completed! Results saved to {output_file}")


audio_path = './audio/'  # 音频文件夹路径
text_path = './text/audio2text.csv'  # 输出文件路径

audio2text(audio_path, text_path)
