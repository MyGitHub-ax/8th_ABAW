import os
import subprocess
import face_alignment
import ffmpeg
import shutil
import config
from toolkit.utils.functions import *
from tqdm import tqdm
def data_augment(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    aug_preprocess(args)

def aug_preprocess(args):
    ## output path
    save_root = args.save_root
    save_video = os.path.join(save_root, 'video')
    save_audio = os.path.join(save_root, 'audio')
    save_face = os.path.join(save_root, 'face')
    save_label = os.path.join(save_root, 'label')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)
    if not os.path.exists(save_label): os.makedirs(save_label)
    if not os.path.exists(save_audio): os.makedirs(save_audio)
    if not os.path.exists(save_face): os.makedirs(save_face)

    ##切分视频，并生成新的label
    cut_video(args.save_root, args.dataset, args.split_length, args.overlap)
    remake_label(args.save_root, args.dataset, args.label_type)

    ##提取音频，人脸，文本
    extract_element(args.save_root,audio=True,face=True,transcription=True)



def cut_video(save_root, dataset, window_length, overlap):
    for file in tqdm(os.listdir(config.PATH_TO_VIDEO[dataset])):
        input_file = os.path.join(config.PATH_TO_VIDEO[dataset], file)
        # Get the duration of the input video
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_file],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        duration = float(result.stdout)

        start = 0
        index = 1
        while start < duration:
            output_file = os.path.join(save_root, 'video', f"{file.split('.')[0]}_{index}.{file.split('.')[1]}")
            if start + window_length < duration:
                command = [
                    "ffmpeg","-y","-v", "quiet", "-i", input_file, "-ss", str(start),
                    "-t", str(window_length), "-c", "copy", output_file
                ]
            else:
                command = [
                    "ffmpeg","-y","-v", "quiet", "-i", input_file, "-ss", str(start),
                    "-t", str(duration - start), "-c", "copy", output_file
                ]
            subprocess.run(command)
            index += 1
            start += (window_length - overlap)


def remake_label(save_root, dataset, label_type):
    label_path = os.path.join(config.PATH_TO_LABEL[dataset], 'partition_orignal.npz')
    label = np.load(label_path, allow_pickle=True)
    label_save = os.path.join(save_root, 'label', 'partition_aug.npz')

    train_partition = label['train'].item()
    val_partition = label['val'].item()
    test_partition = label['test'].item()

    video_path = os.path.join(save_root, 'video')
    new_partition = {}
    if label_type == 'utt':
        for name, partition in [('train', train_partition),
                                ('val', val_partition),
                                ('test', test_partition)]:
            new_partition[name] = {}
            for key in partition.keys():
                video_files = glob.glob(os.path.join(video_path, key + '*'))
                for video_file in video_files:
                    new_partition[name][os.path.basename(video_file.split('.')[0])] = partition[key]
    elif label_type == 'fra':
        for name, partition in [('train', train_partition),
                                ('val', val_partition),
                                ('test', test_partition)]:
            for key in partition.keys():
                video_files = glob.glob(os.path.join(video_path, key + '*'))
                for video_file in video_files:
                    print('施工中！！！！')

    np.savez_compressed(os.path.join(label_save),
                        train=new_partition['train'],
                        val=new_partition['val'],
                        test=new_partition['test'])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Params for preprocess
    parser.add_argument('--split_length', type=int, default=300)
    parser.add_argument('--overlap', type=int, default=120)
    parser.add_argument('--dataset', type=str, default='AVEC2013')
    parser.add_argument('--label_type', type=str, default='utt', help='[utt,fra]')
    args=parser.parse_args()

    root=config.DATA_DIR[args.dataset]
    save_root = os.path.join(root, 'PreProcessed', f'len{args.split_length}_o{args.overlap}')
    args.save_root=save_root
    data_augment(args)