import tarfile
import io
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset

class HCRefLoCoDataset(Dataset):
    def __init__(self, dataset_path, split, custom_transforms=None, load_img=True, images_file="images.tar.gz"):
        """
        Initialize the HCRefLoCoDataset class.

        Parameters:
        - dataset_path (str): Path to the dataset directory.
        - split (str): Dataset split, typically "val" or "test".
        - custom_transforms: Custom image transformations to apply.
        - load_img (bool): Whether to load images from the tar file.
        - images_file (str): Name of the images tar file.
        """
        super(HCRefLoCoDataset, self).__init__()
        assert split in ['val', 'test', 'all'], 'split should be val, test or all'
        self.split = split
        self.dataset_path = dataset_path
        self.images_file = images_file
        self.transforms = custom_transforms
        self.load_img = load_img
        self._load_dataset()

    def load_images_from_tar(self, img_path=None):
        '''
        Load images from the tar file.
        args:
        - img_path (str): Path to the images.tar.gz file.
        '''
        images = {}
        img_path=f"{self.dataset_path}/{self.images_file}" if img_path is None else img_path
        with tarfile.open(img_path, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc='Loading images'):
                if member.isfile() and member.name.endswith(('jpg', 'jpeg', 'png', 'webp')):
                    f = tar.extractfile(member)
                    if f:
                        image = Image.open(io.BytesIO(f.read()))
                        # transfer the grayscale image to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images[member.name] = image
        self.images=images

    def _load_dataset(self):
        self.datas = load_dataset(self.dataset_path)
        all_splits = concatenate_datasets([self.datas['val'],self.datas['test']])
        self.datas['all'] = all_splits
        if self.load_img:
            self.load_images_from_tar()
        else:
            self.images = None

    def change_split(self, split):
        assert split in ['val', 'test', 'all'], 'split should be val, test or all'
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
        if(self.images is not None):
            image = self.images[image_name]
            image = self.transforms(image) if self.transforms else image
        else:
            image = None
        return image, data
