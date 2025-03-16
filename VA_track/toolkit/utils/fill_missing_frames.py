import os
import numpy as np
from PIL import Image


def fill_missing_frames(video_folder, label_file, image_size=(112, 112)):
    """
    根据标签文件的行数确定视频帧数，填充缺失的帧为黑帧（零像素）。

    :param video_folder: 视频文件夹路径，包含视频帧。
    :param label_file: 对应的标签文件路径。
    :param image_size: 图像的大小，默认为 (112, 112)，与其他图像相同。
    """
    # 读取标签文件，计算标签行数（去掉第一行的列名）
    with open(label_file, 'r') as f:
        label_lines = f.readlines()[1:]  # 跳过第一行列名
    num_labels = len(label_lines)  # 标签的行数就是视频应有的帧数

    # 获取视频文件夹中的所有帧文件，并排序
    image_files = sorted(os.listdir(video_folder))
    image_files = [f for f in image_files if f.endswith('.jpg')]  # 只保留jpg文件

    # 获取现有帧的编号列表（保持文件名的原始格式）
    frame_numbers = [f.split('.')[0] for f in image_files]  # 提取文件名（例如 0001, 00286）

    # 计算缺失的帧
    all_frames = set(f"{i:0{len(frame_numbers[0])}d}" for i in range(1, num_labels + 1))  # 使用与现有帧相同的位数
    existing_frames = set(frame_numbers)
    missing_frames = all_frames - existing_frames

    # 填充缺失的帧为黑帧
    for missing_frame in missing_frames:
        missing_filename = f"{missing_frame}.jpg"  # 保持与原图像相同的位数
        missing_filepath = os.path.join(video_folder, missing_filename)

        # 创建黑帧（全零图像，尺寸为 112x112x3）
        black_frame = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)  # RGB黑帧
        black_image = Image.fromarray(black_frame)

        # 保存黑帧为缺失的图像
        black_image.save(missing_filepath)
        print(f"Added black frame: {missing_filename}")


# 遍历数据集中的每个视频文件夹，处理缺失的帧
def fill_missing_frames_in_dataset(dataset_root, label_folder):
    """
    遍历数据集中的每个视频文件夹，检查并填充缺失的帧。

    :param dataset_root: 数据集根目录，其中包含多个视频文件夹。
    :param label_folder: 标签文件夹，包含每个视频对应的标签文件。
    """
    # 获取数据集中的所有视频文件夹
    video_folders = sorted(os.listdir(dataset_root))
    video_folders = [folder for folder in video_folders if os.path.isdir(os.path.join(dataset_root, folder))]

    # 遍历每个视频文件夹，处理缺失的帧
    for video_folder in video_folders:
        video_folder_path = os.path.join(dataset_root, video_folder)

        # 获取对应的视频标签文件路径
        label_file = os.path.join(label_folder, f"{video_folder}.txt")

        # 确保标签文件存在
        if not os.path.exists(label_file):
            print(f"Label file for {video_folder} does not exist, skipping...")
            continue

        print(f"Processing video: {video_folder}")
        fill_missing_frames(video_folder_path, label_file)


# 使用示例
dataset_root = '/data/emotion-data/abaw/PreProcessed/Original/cropped-aligned'  # 设置你的数据集根目录，包含视频文件夹
label_folder = '/data/emotion-data/abaw/PreProcessed/Original/VA_Estimation_Challenge/Train_Set'  # 设置标签文件夹路径
fill_missing_frames_in_dataset(dataset_root, label_folder)
