import numpy as np
import os

# 加载 .npy 文件
dir = "/data/yanglongjiang/project/ABAW/saved-trimodal/result/cv_features:bert-o_800s_1000-FRA+clip-vit-large-patch14-o_800s_1000-FRA+vggish-o_800s_1000-FRA_dataset:ABAW_model:attention_seq+frm_align+None_20250313_220638.npz"
video_path = os.path.join(dir,"video128.npy")
logmel_path = os.path.join(dir,"logmel.npy")
egemaps_path = os.path.join(dir,"egemaps.npy")
VA_continuous_label_path = os.path.join(dir,"VA_continuous_label.npy")
vggish_path = os.path.join(dir,"vggish.npy")
bert_path = os.path.join(dir,"bert.npy")
path_dict = {'video.npy':video_path,'vggish.npy':vggish_path,'bert.npy':bert_path,'VA_continuous_label.npy':VA_continuous_label_path,'egemaps.npy':egemaps_path,'logmel.npy':logmel_path}
# for key,value in path_dict.items():
#     npy_file = np.load(value)
#     print(f"{key} Shape of the array: {npy_file.shape}")
# 验证文件完整性
data = np.load(dir)
print(data)
# print("Keys:", list(data.keys()))          # 应包含 ['frames', 'annotated_index']
# print("Frames shape:", data['frames'].shape)  # 检查维度一致性
# print("Index sample:", data['annotated_index'][:5])  # 查看前5个索引
# data.close()