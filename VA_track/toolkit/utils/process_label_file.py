import os


def process_label_file(input_file, output_file, step=15):
    """
    处理标签文件，每隔 'step' 行保留一个标签，其他的删除。

    :param input_file: 输入标签文件路径
    :param output_file: 输出标签文件路径
    :param step: 保留标签的步长，默认为每隔15行保留一个标签
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 保留第一行（列名）以及每隔 15 行的标签
    header = lines[0]
    filtered_lines = [header]  # 首先保留列名
    for i in range(1, len(lines)):
        if i % step == 0:  # 每隔 15 行保留一个
            filtered_lines.append(lines[i])

    # 保存处理后的标签到新文件
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)


def process_all_label_files(input_dir, output_dir, step=15):
    """
    处理目录下的所有标签文件，每个文件每隔 'step' 行保留一个标签，其他删除。

    :param input_dir: 输入目录，包含多个标签文件
    :param output_dir: 输出目录，用于保存处理后的标签文件
    :param step: 保留标签的步长，默认为每隔15行保留一个标签
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有 .txt 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            # 处理标签文件
            process_label_file(input_file, output_file, step)
            print(f"Processed {filename}")


# 示例使用
input_directory = '/data/emotion-data/abaw/PreProcessed/Original/VA_Estimation_Challenge/Train_Set'  # 原始标签文件所在的目录
output_directory = '/data/emotion-data/abaw/PreProcessed/Original/new_labels/Train_Set'  # 处理后的标签文件保存的目录

# 处理所有文件，每隔 15 行保留一个标签
process_all_label_files(input_directory, output_directory, step=15)
