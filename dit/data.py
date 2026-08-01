"""ImageNet loading from the HuggingFace Hub.

Images are mapped to [-1, 1], which is what the diffusion process expects.

The default repo is ``ILSVRC/imagenet-1k``, which is gated: accept the terms
on the dataset page and authenticate once with ``huggingface-cli login`` (or
set ``HF_TOKEN``). Ungated pre-resized mirrors work too, e.g.
``benjamin-paine/imagenet-1k-256x256`` or ``timm/imagenet-1k-wds``.
"""

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms

DEFAULT_REPO = "ILSVRC/imagenet-1k"

# Column names vary between ImageNet mirrors on the Hub.
IMAGE_KEYS = ("image", "jpg", "img", "png")
LABEL_KEYS = ("label", "cls", "fine_label", "labels")


def center_crop_arr(pil_image, image_size):
    """Center crop as used by ADM/DiT: box-downsample by 2 until close, then crop.

    This preserves more detail than a single bicubic resize when the source
    image is much larger than the target, which is the common case for
    ImageNet at 256x256.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ImageNetTransform:
    """PIL image -> float tensor in [-1, 1] of shape (3, image_size, image_size)."""

    def __init__(self, image_size, random_flip=True):
        self.image_size = image_size
        self.random_flip = random_flip
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True)

    def __call__(self, pil_image):
        arr = center_crop_arr(pil_image.convert("RGB"), self.image_size)
        if self.random_flip and np.random.rand() < 0.5:
            arr = arr[:, ::-1]
        x = self.to_tensor(np.ascontiguousarray(arr))
        return self.normalize(x)


def _find_key(keys, candidates, what):
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"could not find a {what} column among {list(keys)}")


def _num_classes(hf_dataset, label_key):
    feature = hf_dataset.features.get(label_key) if hf_dataset.features else None
    num = getattr(feature, "num_classes", None)
    return num if num is not None else 1000


class HFImageDataset(Dataset):
    """Map-style wrapper over a downloaded HuggingFace image dataset."""

    def __init__(self, hf_dataset, transform, image_key, label_key):
        self.ds = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        x = self.transform(row[self.image_key])
        return x, int(row[self.label_key])


class HFStreamingImageDataset(IterableDataset):
    """Iterable wrapper for streaming mode (no full download needed).

    Shards are split across DataLoader workers so each sample is seen once
    per epoch. Shuffling uses a shuffle buffer on the HF side.
    """

    def __init__(
        self, hf_dataset, transform, image_key, label_key, shuffle_buffer=10_000, seed=0
    ):
        self.ds = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        ds = self.ds
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed + self.epoch, buffer_size=self.shuffle_buffer)
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
        for row in ds:
            yield self.transform(row[self.image_key]), int(row[self.label_key])


def build_dataset(
    repo=DEFAULT_REPO,
    split="train",
    image_size=256,
    random_flip=True,
    streaming=False,
    cache_dir=None,
    seed=0,
):
    """Return ``(dataset, num_classes)`` for an ImageNet-style Hub dataset."""
    hf_dataset = load_dataset(
        repo, split=split, streaming=streaming, cache_dir=cache_dir
    )
    keys = hf_dataset.features.keys()
    image_key = _find_key(keys, IMAGE_KEYS, "image")
    label_key = _find_key(keys, LABEL_KEYS, "label")
    num_classes = _num_classes(hf_dataset, label_key)

    transform = ImageNetTransform(image_size, random_flip)
    if streaming:
        dataset = HFStreamingImageDataset(
            hf_dataset, transform, image_key, label_key, seed=seed
        )
    else:
        # Decode images lazily; PIL objects are produced only on __getitem__.
        dataset = HFImageDataset(hf_dataset, transform, image_key, label_key)
    return dataset, num_classes


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, labels
