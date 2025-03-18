import pickle
import numpy as np
import ast


def count_files(file_path):

    file_count = {}
    try:

        with open(file_path, 'r', encoding='utf-8') as file:

            for line in file:

                line = line.strip()

                items = line.split()
                for item in items:

                    parts = item.split(',')

                    file_prefix = parts[0].split('/')[0]
                    if file_prefix in file_count:

                        file_count[file_prefix] += 1
                    else:

                        file_count[file_prefix] = 1
    except FileNotFoundError:
        print(f"{file_path} not exist")
    return file_count



file_path = r''
counts = count_files(file_path)
counts.pop('image_location')
print(counts)

with open(r"D:\aaa\pre\prediction.pkl", "rb") as f:
    data = pickle.load(f)

output_dict = {}

j = 0
for x in data.keys():
    # print(data[x]['labels'].shape)
    labels = np.argmax(data[x]['logits'], axis=1)
    j += 1
    # print(j)
    print(j)

    if j < 10:
        print(counts['0' + str(j)])
        c = int(counts['0' + str(j)])
    else:
        c = int(counts[str(j)])
        print(c)
    labels = labels[:c]
    print(labels.shape)

    for i, label in enumerate(labels):
        image_path = f"{x.split('/')[1]}/{i + 1:05d}" + ".jpg"
        # print(image_path)
        output_dict[image_path] = label

with open(r'D:\aaa\pre\txt\predictions.txt', 'w') as file:
    file.write(
        "image_location,Fearfully_Surprised,Happily_Surprised,Sadly_Surprised,Disgustedly_Surprised,Angrily_Surprised,Sadly_Fearful,Sadly_Angry\n")

    for key, value in output_dict.items():

        file.write(f'{key},{value}\n')

