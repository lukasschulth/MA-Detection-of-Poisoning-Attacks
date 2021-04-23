import os
import PIL.Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TrafficSignDataset(Dataset):

    #def __init__(self, data_dir, transform: transforms):
    def __init__(self, data_dir, transform: transforms):
        # Get your Dataset here

        self.dataset_dictionary = self.load_data_dir(data_dir=data_dir)
            #self.images = self.resizeImages(images=self.images)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_dictionary['images'])

    def __getitem__(self, idx):

        label = self.dataset_dictionary['labels'][idx]
        image = self.dataset_dictionary['images'][idx]
        poison_label = self.dataset_dictionary['poison_labels'][idx]
        path = self.dataset_dictionary['paths'][idx]

        ## Swap  W x H x C to H x W x C
        # image = torch.FloatTensor(image).transpose(1, 0)
        # image = image.transpose(2, 0)

        # image = transforms.ToPILImage()(image)
        # image = self.transform(image)

        if self.transform is not None:
            image = self.transform(image)
        data_dict = {'image': image, 'label': label, 'poison_label': poison_label, 'path': path}
        return data_dict

    def load_data_dir(self, data_dir):
        # Get all subdirectories of data_dir. Each represents a label.
        directories = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))]
        # d verschiedene Labels

        # Loop through the label directories and collect the data in
        # two lists, labels and images.
        labels = []
        images = []
        poison_labels = []
        paths = []

        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if not f.endswith(".csv")]

            for f in file_names:
                if f.endswith("_poison.jpeg"):
                    poison_labels.append(int(1))
                else:
                    poison_labels.append(int(0))
                # images.append(skimage.data.imread(f))
                images.append(PIL.Image.open(f))
                labels.append(int(d))
                paths.append(f)

        dataset_dictionary = {'labels': labels, 'images': images, 'poison_labels': poison_labels, 'paths': paths}

        return dataset_dictionary
