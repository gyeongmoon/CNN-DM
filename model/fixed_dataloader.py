import torch
import torch.multiprocessing as multiprocessing
import torch.utils.data as data
from torchvision.datasets import folder
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
import collections
import sys
import traceback
import threading
import random
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


_use_shared_memory = False
"""Whether to use shared memory in default_collate"""

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=folder.default_loader):
        classes, class_to_idx = folder.find_classes(root)
        imgs = folder.make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(folder.IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idx, random_samples = index
        path, target = self.imgs[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img, random_samples)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self):
        return len(self.imgs)


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed

        torch.manual_seed(self.seed)
        self.curr_state = torch.get_rng_state()

    def __iter__(self):
        torch.set_rng_state(self.curr_state)
        batch_indices = torch.randperm(len(self.data_source)).long()
        self.curr_state = torch.get_rng_state()
        return iter(batch_indices)  # It gives only one value.

    def __len__(self):
        return len(self.data_source)


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices, random_samples = r
        try:
            samples = collate_fn([dataset[i, random_samples[idx_sample * 3:(idx_sample+1) * 3]] for idx_sample, i in enumerate(batch_indices)])  # Call transform(img).
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader, seed=1):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()
        self.seed = seed

        self.sample_iter = iter(self.batch_sampler)

        random.seed(self.seed)
        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue

            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return

        random_samples = []
        for batch_size in range(len(indices) * 3):
            random_samples.append(random.random())

        self.index_queue.put((self.send_idx, indices, random_samples))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, seed=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset, self.seed)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self, seed=self.seed)

    def __len__(self):
        return len(self.batch_sampler)
