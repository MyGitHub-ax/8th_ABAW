import os
from collections import defaultdict


def count_subdirs_recursive(path):
    """递归统计目录下的所有子目录数（包含嵌套）"""
    count = 0
    for entry in os.scandir(path):
        if entry.is_dir():
            count += 1  # 计数当前子目录
            count += count_subdirs_recursive(entry.path)  # 递归统计
    return count

import pickle
def get_subdir_counts(root_path):
    """
    获取每个直接子目录的子目录数量统计

    参数:
    root_path (str): 要扫描的根目录路径

    返回:
    dict: {子目录名: 包含的子目录总数}
    """
    stats = {}

    # 遍历根目录的直接子目录
    for entry in os.scandir(root_path):
        if entry.is_dir():
            dir_name = entry.name
            stats[dir_name] = count_subdirs_recursive(entry.path)

    return stats


if __name__ == "__main__":
    target_path = "/data/liuran/lr/real/feature-vs-text-compound-emotion-main/dataset/C-EXPR-DB-CHALLENGE/cropped_aligned"  # 替换为你的目录路径
    counts = get_subdir_counts(target_path)

    import gzip

    with gzip.open('/data/liuran/data.pkl', 'wb') as f:
        pickle.dump(counts, f)
    with open('/data/liuran/counts.txt', 'w', encoding='utf-8') as f:
        f.write(repr(counts))
    print(counts)
    # 打印格式化结果
    print(f"{'目录名称':<20} | {'子目录数量':<10}")
    print("-" * 32)
    for name, count in counts.items():
        print(f"{name:<20} | {count:<10}")