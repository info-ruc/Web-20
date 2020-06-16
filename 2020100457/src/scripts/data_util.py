from torch.utils.data import Dataset
import torch.utils.data as Data

class ListDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_dataset(path, config, test=False, shuffle=True):
    train_list = []
    label_list = []
    if test: shuffle = False
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            splits = line.split("\t")
            label_list.append(int(splits[1].strip()))
            train_list.append(splits[2].strip())
    
    torch_dataset = ListDataset(train_list, label_list)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,  # 批大小
        shuffle=shuffle,  # 是否随机打乱顺序
        num_workers=4,  # 多线程读取数据的线程数
    )
    return loader