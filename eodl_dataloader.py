import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self,args):
        super(MyDataset, self).__init__()
        dataset = sio.loadmat('./datasets/{}.mat'.format(args.dataset))
        self.data = torch.from_numpy(dataset['data'].astype('float32'))
        
        label = dataset['label'].flatten()
        #get dataset classes number
        self.classes = int(max(label))+1
        self.label = torch.from_numpy(label).to(torch.long)

    def __getitem__(self, item):
        return self.data[item],self.label[item]

    def __len__(self):
        return len(self.data)

    def get_nIn(self):
        return len(self.data[0])

    def get_nOut(self):
        return self.classes

def get_dataset(args):
    dataset = MyDataset(args)
    dataloader = DataLoader(dataset,batch_size=args.bs)
    return dataloader,dataset