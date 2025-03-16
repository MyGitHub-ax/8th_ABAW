import os
import shutil


def reorganize_features(source_dir, target_root, features_to_extract):
    """
    重组特征文件目录结构

    参数：
    source_dir: 源目录路径（包含视频子目录）
    target_root: 目标根目录路径（特征目录的父级目录）
    features_to_extract: 要提取的特征文件名列表（如 ["vggish.npy"]）
    """
    # 遍历视频目录
    for video_name in os.listdir(source_dir):
        video_path = os.path.join(source_dir, video_name)

        if not os.path.isdir(video_path):
            continue

        # 处理目标特征文件
        for feature_file in features_to_extract:
            src_path = os.path.join(video_path, feature_file)

            if not os.path.exists(src_path):
                continue

            # 构建目标路径
            feature_name = os.path.splitext(feature_file)[0]
            target_dir = os.path.join(target_root, feature_name)
            os.makedirs(target_dir, exist_ok=True)

            # 移动并重命名文件
            new_filename = f"{video_name}.npy"
            dst_path = os.path.join(target_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")


if __name__ == "__main__":
    # 配置参数
    SOURCE_DIR = "/data/emotion-data/abaw/PreProcessed/RJCMA/compacted_48"  # 源特征目录
    TARGET_ROOT = "/data/emotion-data/abaw/PreProcessed/RJCMA/PreProcessed/Original/compacted_48"  # 目标根目录
    FEATURES_TO_EXTRACT = ["bert.npy"]  # 要提取的特征文件列表

    reorganize_features(SOURCE_DIR, TARGET_ROOT, FEATURES_TO_EXTRACT)