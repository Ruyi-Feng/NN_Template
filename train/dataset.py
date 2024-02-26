from torch.utils.data import Dataset


class My_Dataset(Dataset):
    def __init__(self):
        """
        用于初始化数据与设置
        """
        pass

    def __getitem__(self, index: int):
        """
        用于获取数据,index为索引
        """
        pass
        return data[index]

    def __len__(self):
        """
        用于获取数据长度
        """
        pass
        return len(data)
