import os.path
from charset_normalizer import from_path
import config
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

def split_feature(feature_path, save_feature, args):
    if not os.path.exists(save_feature): os.makedirs(save_feature)
    for file in tqdm.tqdm(sorted(os.listdir(os.path.join(feature_path, args.feature)))):
        feature = np.load(os.path.join(feature_path, args.feature, file))
        length = feature.shape[0]
        split_length = args.split_length
        overlap = args.overlap
        stride = split_length - overlap  # 步长# 计算可以获得的窗口数

        i = 0
        while i * stride < length:
            start = i * stride
            end = min(start + split_length, length)
            window_feature = feature[start:end]

            if os.path.exists(os.path.join(save_feature, file[:-4] + '_' + str(i) + '.npy')):
                print('feature already exists')
            else:
                np.save(os.path.join(save_feature, file[:-4] + '_' + str(i) + '.npy'), window_feature)
            i += 1

def split_label(file,label_path, dir, new_label_save_path, args):
    names = []
    labels = []
    with open(os.path.join(label_path, dir, file), 'r') as f:
        label = f.readlines()
    label = label[1:]
    label = [[float(j) for j in i.split(',')] for i in label]
    length = len(label)
    split_length = args.split_length
    overlap = args.overlap
    stride = split_length - overlap  # 步长
    num_windows = (length - split_length) // stride + 1  # 计算可以获得的窗口数
    for i in range(num_windows):
        start = i * stride  # 窗口的起始位置
        end = start + split_length  # 窗口的结束位置
        window_label = label[start:end]  # 截取label
        # 添加到names和labels列表中
        names.append(file[:-4] + '_' + str(i))
        labels.append(window_label)
    for name, label in zip(names, labels):
        if os.path.exists(os.path.join(new_label_save_path, name + '.txt')):
            print("label exist!")
        else:
            with open(os.path.join(new_label_save_path, name + '.txt'), 'w') as f:
                # write f as valance,arousal\nxx,xx\nxx,xx\n
                f.write('valance,arousal\n')
                for i in label:
                    f.write(str(i[0]) + ',' + str(i[1]) + '\n')
            f.close()

def split_label_feature_by_length_ABAW(data_root,save_root,args):
    feature_path=config.PATH_TO_FEATURES['ABAW']
    label_path = config.PATH_TO_LABEL['ABAW']

    ## output path
    save_split_feature = os.path.join(save_root,'features')
    save_split_label = os.path.join(save_root,'labels')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_split_feature): os.makedirs(save_split_feature)
    if not os.path.exists(save_split_label): os.makedirs(save_split_label)


    feature_name = args.feature.split('-')
    if args.pseudo:
        new_feature_name = ('-'.join(feature_name[:-1]) + '-' + 'o_'+str(args.overlap)+'s_'+str(args.split_length)
                            + '-'+'pseudo'+'-' + feature_name[-1])
    else:
        new_feature_name = '-'.join(feature_name[:-1]) + '-' + 'o_' + str(args.overlap) + 's_' + str(
            args.split_length) + '-' + feature_name[-1]
    save_feature = os.path.join(save_split_feature, new_feature_name)

    #切割特征
    print('split_feature_path:',data_root)
    # split_feature(feature_path, save_feature, args)

    #切割标签
    for dir in tqdm.tqdm(sorted(os.listdir(label_path))):
        if args.pseudo:
            label_name = 'o_'+str(args.overlap)+'s_'+str(args.split_length)+'_pseudo'
        else:
            label_name = 'o_'+str(args.overlap)+'s_'+str(args.split_length)

        new_label_save_path = os.path.join(save_split_label, label_name, dir)
        if not os.path.exists(new_label_save_path):
            os.makedirs(new_label_save_path)

        for file in sorted(os.listdir(os.path.join(label_path, dir))):

            if args.test_no_label and dir == 'Test_Set':
                Test_files = sorted([feature_file for feature_file in os.listdir(save_feature) if feature_file.split('_')[0]==file[:-4]],key=lambda x:int(x.split('_')[-1].split('.')[0]))
                step=int(args.split_length/(args.split_length-args.overlap))
                for Test_file in Test_files[::step]:
                    if os.path.exists(os.path.join(new_label_save_path, Test_file[:-4] + ".txt")):
                        print('label already exists')
                    else:
                        label_file = open(os.path.join(new_label_save_path, Test_file[:-4] + ".txt"), "w")
                        label_file.close()
            elif dir == 'Pseudo_Test_Set':
                if not args.pseudo:
                    continue
                else:
                    split_label(file, 'Train_set', label_path, new_label_save_path, args)

            else:
                split_label(file, label_path, dir, new_label_save_path, args)

def read_train_val_test(label_path, data_type,args):
    names, labels = [], []
    # if args.pseudo:
    #     label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)+'_pseudo'
    # else:
    #     label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)
    assert data_type in ['train', 'val', 'test']
    # videoIDs, videoLabels, _, _, trainVids, valVids, testVids = pickle.load(open(label_path, "rb"), encoding='utf-8')
    if data_type == 'train':
        if args.pseudo:
            for file in sorted(os.listdir(os.path.join(label_path, 'Train_Set'))):
                with open(os.path.join(label_path, 'Train_Set', file), 'r') as f:
                    label = f.readlines()
                label = label[1:]
                label = [[float(j) for j in i.split(',')] for i in label]
                names.append(file[:-4])
                labels.append(label)
        else:
            for file in sorted(os.listdir(os.path.join(label_path, 'Train_Set'))):
                with open(os.path.join(label_path, 'Train_Set', file), 'r') as f:
                    label = f.readlines()
                label = label[1:]
                label = [[float(j) for j in i.split(',')] for i in label]
                names.append(file[:-4])
                labels.append(label)
    if data_type == 'val':
        for file in sorted(os.listdir(os.path.join(label_path, 'Validation_Set'))):
            with open(os.path.join(label_path, 'Validation_Set', file), 'r') as f:
                label = f.readlines()
            label = label[1:]
            label = [[float(j) for j in i.split(',')] for i in label]
            names.append(file[:-4])
            labels.append(label)
    if data_type == 'test':
        for file in sorted(os.listdir(os.path.join(label_path, 'Test_Set'))):
            with open(os.path.join(label_path, 'Test_Set', file), 'r') as f:
                label = f.readlines()
            label = label[1:]
            label = [[float(j) for j in i.split(',')] for i in label]
            names.append(file[:-4])
            labels.append(label)
    return names, labels


def normalize_dataset_format(data_root,save_root,args):
    # gain paths

    ## output path
    save_split_label = os.path.join(save_root,'labels')

    split_label_feature_by_length_ABAW(data_root, save_root,args)


    # # gain (names, labels)
    train_names, train_labels = read_train_val_test(data_root, 'train',args)
    val_names, val_labels = read_train_val_test(data_root, 'val',args)
    test_names, test_labels = read_train_val_test(data_root, 'test',args)
    if args.shuffle:
        train_data = list(zip(train_names, train_labels))
        val_data = list(zip(val_names, val_labels))
        random.shuffle(train_data)
        random.shuffle(val_data)

        # 计算划分比例
        train_size = int(len(train_data) * 0.75)
        val_size = len(train_data) - train_size

        # 划分测试集和验证集
        train_set = train_data[:train_size]
        val_set = train_data[train_size:]

        # 解压缩样本和标签
        train_names, train_labels = zip(*train_set)
        val_names, val_labels = zip(*val_set)

    print(f'train: {len(train_names)}')
    print(f'val:   {len(val_names)}')
    print(f'test:  {len(test_names)}')

    ## generate label path
    whole_corpus = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val', val_names, val_labels),
                                     ('test', test_names, test_labels)]:
        whole_corpus[name] = {}
        for ii, videoname in enumerate(videonames):
            whole_corpus[name][videoname] = {'emo': 0, 'val': labels[ii]}

    if args.shuffle:
        random_number = random.randint(1, 1000)  # 生成一个范围在1到1000之间的随机整数
        random_number_str = str(random_number).zfill(4)  # 将随机整数转换为字符串，并在左侧用0填充至总长度为4位
        if args.pseudo:
            np.savez_compressed(os.path.join(save_split_label,'npz', f'label_{random_number_str}_pseudo.npz'),
                            train_corpus=whole_corpus['train'],
                            val_corpus=whole_corpus['val'],
                            test_corpus=whole_corpus['test'])
        else:
            np.savez_compressed(os.path.join(save_split_label,'npz',  f'label_{random_number_str}.npz'),
                                train_corpus=whole_corpus['train'],
                                val_corpus=whole_corpus['val'],
                                test_corpus=whole_corpus['test'])
    else:
        np.savez_compressed(os.path.join(save_split_label,'npz',  'label.npz'),
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])

def rename_args_feature(args,feature_name,m):
    feature_name = feature_name.split('-')
    if args.pseudo:
        new_feature_name = ('-'.join(feature_name[:-1]) + '-' + 'o_'+str(args.overlap)+'s_'+str(args.split_length)
                            + '-'+'pseudo'+'-' + feature_name[-1])
    else:
        new_feature_name = '-'.join(feature_name[:-1]) + '-' + 'o_' + str(args.overlap) + 's_' + str(
            args.split_length) + '-' + feature_name[-1]
    if m=='a':
        args.audio_feature=new_feature_name
    elif m=='v':
        args.video_feature=new_feature_name
    else:
        args.text_feature=new_feature_name
    return args
def abaw_dataprocess(args):
    feature_list=[args.audio_feature,args.text_feature,args.video_feature]
    feature_type=['a','v','t']
    if args.pseudo:
        args.shuffle=True
    for feature,type in zip(feature_list,feature_type):
        args.feature=feature

        split_label_feature_by_length_ABAW(args)
        args=rename_args_feature(args,feature,type)

    normalize_dataset_format(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_length', type=int, default=1000)
    parser.add_argument('--overlap', type=int, default=800)
    parser.add_argument('--feature',type=str, default='vggish-FRA')
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--pseudo',action='store_true')
    parser.add_argument('--test_no_label',action='store_true')
    args=parser.parse_args()
    if args.pseudo:
        args.shuffle=True
    data_root = '/data/emotion-data/abaw/PreProcessed/RJCMA'
    # save_root = '/data/wenzhuofan/Data/ABAW/Split_Feature_Label'
    save_root = '/data/emotion-data/abaw/PreProcessed/RJCMA/PreProcessed/Original/new_labels'
    normalize_dataset_format(data_root,save_root,args)
