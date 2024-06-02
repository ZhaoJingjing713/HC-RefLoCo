import tarfile
import io
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

class HCRefLoCoDataset(Dataset):
    def __init__(self, dataset_path, split, custom_transforms=None):
        """
        Initialize the HCRefLoCoDataset class.

        Parameters:
        - dataset_path (str): Path to the dataset directory.
        - split (str): Dataset split, typically "val" or "test".
        - custom_transforms: Custom image transformations to apply.
        """
        super(HCRefLoCoDataset, self).__init__()
        assert split in ['val', 'test'], 'split should be val or test'
        self.split = split
        self.dataset_path = dataset_path
        self.images_file = "images.tar.gz"
        self.transforms = custom_transforms
        self._load_dataset()

    def _load_images_from_tar(self):
        images = {}
        with tarfile.open(f"{self.dataset_path}/{self.images_file}", "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(('jpg', 'jpeg', 'png', 'webp')):
                    f = tar.extractfile(member)
                    if f:
                        image = Image.open(io.BytesIO(f.read()))
                        # transfer the grayscale image to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images[member.name] = image
        return images

    def _load_dataset(self):
        self.datas = load_dataset(self.dataset_path)
        self.images = self._load_images_from_tar()

    def change_split(self, split):
        assert split in ['val', 'test'], 'split should be val or test'
        self.split = split

    def __len__(self):
        return len(self.datas[self.split])

    def __getitem__(self, idx):
        """
        Returns:
        - image (Tensor): Transformed image data.
        - data (dict): Other sample data.
        """
        data = self.datas[self.split][idx]
        image_name = data['file_name']
        image = self.images[image_name]
        image = self.transforms(image) if self.transforms else image
        return image, data
