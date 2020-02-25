import os

folder_mapper = {'0': '0',
                 '1': '1',
                 '2': '2',
                 '3': '3',
                 '4': '4',
                 '5': '5',
                 '6': '6',
                 '7': '7',
                 '8': '8',
                 '9': '9',
                 'A': 'A',
                 'B': 'B',
                 'C': 'C',
                 'cuan': '川',
                 'D': 'D',
                 'E': 'E',
                 'e1': '鄂',
                 'F': 'F',
                 'G': 'G',
                 'gan': '赣',
                 'gan1': '甘',
                 'gui': '贵',
                 'gui1': '桂',
                 'H': 'H',
                 'hei': '黑',
                 'hu': '沪',
                 'J': 'J',
                 'ji': '冀',
                 'ji1': '吉',
                 'jin': '津',
                 'jing': '京',
                 'K': 'K',
                 'L': 'L',
                 'liao': '辽',
                 'lu': '鲁',
                 'M': 'M',
                 'meng': '蒙',
                 'min': '闽',
                 'N': 'N',
                 'ning': '宁',
                 'P': 'P',
                 'Q': 'Q',
                 'qing': '青',
                 'qiong': '琼',
                 'R': 'R',
                 'S': 'S',
                 'shan': '陕',
                 'su': '苏',
                 'sx': '晋',
                 'T': 'T',
                 'U': 'U',
                 'V': 'V',
                 'W': 'W',
                 'wan': '皖',
                 'X': 'X',
                 'xiang': '湘',
                 'xin': '新',
                 'Y': 'Y',
                 'yu': '豫',
                 'yu1': '渝',
                 'yue': '粤',
                 'yun': '云',
                 'Z': 'Z',
                 'zang': '藏',
                 'zhe': '浙'}


# 生成车牌字符图像列表
def gen_dataset():
    data_path = 'train'
    character_folders = os.listdir(data_path)
    label = 0
    train_dataset = []
    valid_dataset = []
    character_mapper = {}
    for character_folder in character_folders:
        if character_folder == '.DS_Store' \
                or character_folder == '.ipynb_checkpoints':
            continue
        character_mapper[label] = folder_mapper.get(character_folder, '')
        # print(str(label) + "  " + character_folder + "  " + character_mapper[label])
        character_imgs = os.listdir(os.path.join(data_path, character_folder))
        for i in range(len(character_imgs)):
            if i % 10 == 0:
                valid_dataset.append((os.path.join(os.path.join(data_path, character_folder),
                                                   character_imgs[i]), label))
            else:
                train_dataset.append((os.path.join(os.path.join(data_path, character_folder),
                                                   character_imgs[i]), label))
        label = label + 1
    return train_dataset, valid_dataset, character_mapper


if __name__ == '__main__':
    train_dataset, valid_dataset, character_mapper = gen_dataset()
    print('train size:{}, valid_size:{}, mapper_len:{}'.format(len(train_dataset),
                                                               len(valid_dataset),
                                                               len(character_mapper)))
