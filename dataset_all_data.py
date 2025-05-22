import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


class ApneaDataset_all(Dataset):
    def __init__(self, data_folders, length = 30000): #30000
        self.data_folders = data_folders

        self.length = length
        self.a_items = []
        self.na_items = []
        for root, _, files in os.walk(data_folders, topdown=False):
            for name in files:
                if os.path.isfile(os.path.join(root, name)):
                    if name.endswith('pth'):
                        if 'apnea' in os.path.join(root, name).split('/'):
                            self.a_items.append(os.path.join(root, name))
                        elif 'nonapnea' in os.path.join(root, name).split('/'):
                            self.na_items.append(os.path.join(root, name))
        print(f'found {len(self.a_items)} apnea files') 
        print(f'found {len(self.na_items)} non apnea files') 
    

    def __len__(self):
        
        return self.length
    

    def __getitem__(self, idx):
        # Activate shuffle to see entire (real) dataset
        random.shuffle(self.a_items)
        random.shuffle(self.na_items)
        
        a1_idx = idx%len(self.a_items)
        a2_idx = random.randint(0,len(self.a_items)-1)

        na1_idx = idx//len(self.a_items)
        na2_idx = random.randint(0,len(self.na_items)-1)

        pn = random.randint(0,3)
        # generate positive (A-A or NA-NA) pair, or negative (A-NA or NA-A) pair, with equal probability
        if pn==0:
            # A-A
            S_1 = torch.load(self.a_items[a1_idx])
            lbl_1 = 1
            S_2 = torch.load(self.a_items[a2_idx])
            lbl_2 = 1
        elif pn==1:
            # NA-NA
            S_1 = torch.load(self.na_items[na1_idx])
            lbl_1 = 0
            S_2 = torch.load(self.na_items[na2_idx])
            lbl_2 = 0
        elif pn==2:
            # A-NA
            S_1 = torch.load(self.a_items[a1_idx])
            lbl_1 = 1
            S_2 = torch.load(self.na_items[na2_idx])
            lbl_2 = 0
        else:
            # NA-A
            S_1 = torch.load(self.na_items[na1_idx])
            lbl_1 = 0
            S_2 = torch.load(self.a_items[a2_idx])
            lbl_2 = 1

        # normalizations
        # S += 50
        # S /= 50
        
        return S_1.unsqueeze(0), S_2.unsqueeze(0), lbl_1, lbl_2

if __name__ == '__main__':
    
    data_path = "/disks/disk1/dlillini/PSG_Audio/MELSP_6S/train"
    dset = ApneaDataset_all(data_path)
    dload = DataLoader(dset, batch_size=256, shuffle=True)

    print(len(dload))
    for n, (S_1, S_2, lbl_1, lbl_2) in enumerate(dload):
         print(f'{n} - {S_1.shape} - {S_2.shape}')