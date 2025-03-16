import shutil

import config
from toolkit.utils.functions import *
import ffmpeg
import cv2
import os
import face_alignment
import torch
from tqdm import tqdm

# 自定义读取写在这个函数内，
# 目标：对应partition的VideoPathList以及LabelList
def read_train_val_test(data_root, data_type):
    names, labels, video_paths = [], [], []
    assert data_type in ['train', 'val', 'test']
    ################################从这开始写####################################
    label_path=os.path.join(data_root,'Depression_label',data_type+'_label')
    video_path=os.path.join(data_root,'train' if data_type=='train' else 'dev' if data_type=='val' else 'test')

    #自定义排序方法
    def sort_key(filename):
        # 按照'_'分割，取第二个'_'之前的部分
        parts = filename.split('_')
        assert len(parts) > 1
        new_filename=parts[0]+'_'+parts[1]
        return new_filename

    for f in sorted(os.listdir(label_path),key=sort_key):
        with open(os.path.join(label_path,f),'r') as label_file:
            for line in label_file:
                label = int(line.strip())
            labels.append(label)

    for f in sorted(os.listdir(video_path)):
        video_paths.append(os.path.join(video_path, f))

    ############################################################################
    if all(not isinstance(i, list) for i in labels):  # 如果所有元素都不是列表，则为一维
        labels = [[i] for i in labels]  # 给每个元素增加一个维度

    for f in video_paths:
        f=os.path.basename(f)
        names.append(f.split('.')[0])
    return names, labels, video_paths

def extract_element(data_root, audio=False, face=False, transcription=False,audio_sr=16000):
    for f in tqdm(os.listdir(os.path.join(data_root,'video'))):
        f_name=f.split('.')[0]
        f_path=os.path.join(data_root,'video', f)
        # extract audio
        if audio:
            (
            ffmpeg
            .input(f_path)
            .output(os.path.join(data_root,'audio',f_name+'.wav'), ar=audio_sr, ac=1)
            .run(overwrite_output=True,quiet=True)
            )
        # extract face
        if face:
            cap = cv2.VideoCapture(f_path)

            # 初始化face-alignment的检测器
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                              device='cuda' if torch.cuda.is_available() else 'cpu')
            frame_count = 0
            target_face = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                # 检测人脸并对齐
                try:
                    landmarks = fa.get_landmarks(frame)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
                if landmarks is not None:
                    for i, landmark in enumerate(landmarks):
                        # 获取人脸边界框
                        x_min = int(min(landmark[:, 0]))
                        x_max = int(max(landmark[:, 0]))
                        y_min = int(min(landmark[:, 1]))
                        y_max = int(max(landmark[:, 1]))
                        # 裁剪人脸
                        face_frame = frame[y_min:y_max, x_min:x_max]
                        try:
                            target_face[i].append(face_frame)
                        except:
                            target_face.append([face_frame])
            os.makedirs(os.path.join(data_root,'face',f_name), exist_ok=True)
            if target_face == []:
                cap.release()
                cv2.destroyAllWindows()
                continue
            target_face = max(target_face, key=len)
            target_face = [face for face in target_face if face.size > 0]

            for i, frame in enumerate(target_face):
                face_filename = os.path.join(data_root,'face',f_name, f"{i}.jpg")
                cv2.imwrite(face_filename, frame)
            cap.release()
            cv2.destroyAllWindows()
        # extract transcription
        if transcription:
            print('还没写呢！！！')

def normalize_dataset_format(data_root, save_root):
    # gain (names, labels)
    train_names, train_labels, train_video_paths = read_train_val_test(data_root, 'train')
    val_names, val_labels, val_video_paths = read_train_val_test(data_root, 'val')
    test_names, test_labels, test_video_paths = read_train_val_test(data_root, 'test')

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_audio = os.path.join(save_root, 'audio')
    save_face  = os.path.join(save_root, 'face')
    save_label = os.path.join(save_root, 'label')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)
    if not os.path.exists(save_label): os.makedirs(save_label)
    if not os.path.exists(save_audio): os.makedirs(save_audio)
    if not os.path.exists(save_face): os.makedirs(save_face)

    # 把全部video汇总到一个文件夹
    for video_paths in [train_video_paths, val_video_paths, test_video_paths]:
        for video_path in video_paths:
            shutil.copy(video_path, os.path.join(save_video, os.path.basename(video_path)))
    print(f'train number: {len(train_names)}')
    print(f'val   number: {len(val_names)}')
    print(f'test  number: {len(test_names)}')

    ## generate label path
    partition = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val', val_names, val_labels),
                                     ('test', test_names, test_labels)]:
        partition[name] = {}
        for ii, videoname in enumerate(videonames):
            partition[name][videoname] = labels[ii]

    np.savez_compressed(os.path.join(save_label,'partition_orignal.npz'),
                        train=partition['train'],
                        val=partition['val'],
                        test=partition['test'])

    # extract audio, extract faces, extract transcription
    extract_element(save_root,audio=True,face=True,transcription=True)


# run -d toolkit/preprocess/mer2024.py
if __name__ == '__main__':
    DataSet='AVEC2013'
    data_root = os.path.join(config.DATA_DIR[DataSet], 'RAW')
    save_root = os.path.join(config.DATA_DIR[DataSet], 'PreProcessed', 'Original')
    normalize_dataset_format(data_root, save_root)

