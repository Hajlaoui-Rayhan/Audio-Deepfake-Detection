import torch
import pandas
import torch.nn.functional as F
import torchaudio


class CustData(torch.utils.data.Dataset):

    def __init__(self,audio_files, train_folders, train_target):
        self.audio_files = audio_files
        self.train_folders = train_folders
        self.train_target = train_target
        #self.processor = processor

    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, idx):
        audio_name = self.audio_files[idx]

        audio_sample = torchaudio.load(self.train_folders + audio_name)
        #inputs = self.processor(audio_sample, return_tensors="pt", padding=True, truncation=True)

        pd = pandas.read_csv(self.train_target, sep=' ', header=None)




        target = pd.loc[pd[1]+'.pt' == audio_name][5]




        if (target == "spoof").all():
            target = torch.zeros(1)
        else :
            target = torch.ones(1)
        #print(audio_sample)
        return resize_tensor(audio_sample[0]) , target



def resize_tensor(tensor, target_size=10000):
    # Flatten the tensor
    tensor = tensor.flatten()
    current_size = tensor.size(0)

    if current_size < target_size:
        # Pad with zeros if the tensor is smaller than the target size
        padding_size = target_size - current_size
        tensor = F.pad(tensor, (0, padding_size), 'constant', 0)
    elif current_size > target_size:
        # Trim the tensor if it is larger than the target size
        tensor = tensor[:target_size]

    # Reshape to (1, target_size)
    #tensor = tensor.view(1, target_size)
    return tensor