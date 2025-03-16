import os

############ For LINUX ##############

##需要自己配置数据集路径和数据增量文件夹
###############################################################################################
DATA_DIR = {
	'MER2023': '/share/home/lianzheng/chinese-mer-2023/dataset/mer2023-dataset-process',
	'ABAW': '/data/emotion-data/abaw/PreProcessed/RJCMA',
    'ODYSSEY':'/data/emotion-data/Odyssey',
	'AVEC2013':'/data/wenzhuofan/Data/AVEC2013',
}

AUGMENTED_SETTING = 'Original'
###############################################################################################

# 定义一个函数生成路径
def generate_paths(data_dirs,aug_setting, subdirs):
    paths = {dataset: {} for dataset in data_dirs}
    for dataset, base_path in data_dirs.items():
        for subdir_name, subdir_path in subdirs.items():
            paths[dataset][subdir_name] = os.path.join(base_path,'PreProcessed',aug_setting, subdir_path)
    return paths

# 需要生成的子路径
subdirs = {
    'audio': 'Audios',
    'video': 'Videos',
    'face': 'cropped-aligned',
    'transcription': 'transcribe',
    'feature': 'split_feature',
    'label': 'split_label',
}

# 针对不同的数据集进行调整
custom_subdirs = {
    # 'ABAW': {
    #     'audio': 'Audio',
    # },
    # 'AVEC2013': {
    #     'face': 'CA_face',
    # }
}

# 生成所有路径
PATHS = {}
for dataset in DATA_DIR:
    subdir_override = custom_subdirs.get(dataset, {})
    subdir_mapping = {**subdirs, **subdir_override}
    PATHS[dataset] = generate_paths({dataset: DATA_DIR[dataset]},AUGMENTED_SETTING, subdir_mapping)[dataset]

PATH_TO_AUDIO = {dataset: paths['audio'] for dataset, paths in PATHS.items() if 'audio' in paths}
PATH_TO_VIDEO = {dataset: paths['video'] for dataset, paths in PATHS.items() if 'video' in paths}
PATH_TO_FACE = {dataset: paths['face'] for dataset, paths in PATHS.items() if 'face' in paths}
PATH_TO_TRANSCRIPTIONS = {dataset: paths['transcription'] for dataset, paths in PATHS.items() if 'transcription' in paths}
PATH_TO_FEATURES = {dataset: paths['feature'] for dataset, paths in PATHS.items() if 'feature' in paths}
PATH_TO_LABEL = {dataset: paths['label'] for dataset, paths in PATHS.items() if 'label' in paths}


# pre-trained models, including supervised and unsupervised
PATH_TO_PRETRAINED_MODELS = '/data/yanglongjiang/Data/tools'
PATH_TO_OPENSMILE = '/data/yanglongjiang/Data/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/data/yanglongjiang/Data/tools/ffmpeg-4.4.1-i686-static/ffmpeg'

# dir
SAVED_ROOT = os.path.join('./saved')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows [OpenFace to extract face] ##############
DATA_DIR_Win = {
	'MER2023': 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\mer2023-dataset-process',
    'ABAW': 'C:\\Users\\16426\\Documents\\MyDoc\\EN_TRASH\\ABAW_openface',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
    'ABAW':os.path.join(DATA_DIR_Win['ABAW'], 'CA_all'),
	# 'AVEC2013':os.path.join(DATA_DIR_Win['AVEC2013'], 'CA_all'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
    'ABAW':os.path.join(DATA_DIR_Win['ABAW'], 'features'),
}

PATH_TO_OPENFACE_Win = "C:\\Users\\16426\\Documents\\MyDoc\\研究生\\杂货铺\\OpenFace_2.2.0_win_x64"
