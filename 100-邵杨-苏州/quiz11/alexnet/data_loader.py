from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, txt_path, image_path, transform=None):
        fh = open(txt_path, 'r')
        images = []
        for line in fh:
            line = line.rstrip()
            words = line.split(';')
            images.append((words[0], int(words[1])))
            self.images = images
            self.transform = transform
            self.image_path = image_path

    def __getitem__(self, index):
        fn, label = self.images[index]
        img = Image.open(self.image_path + '/' + fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


#img = Image.open(r".\image" + '/' + 'cat.0.jpg').convert('RGB')
#print(img)
