import torch
from torch.utils.data import Sampler, DistributedSampler
from dataset import Dataset
import random
from typing import Optional

class BatchSamplerSimilarLabel(Sampler):
    def __init__(self,dataset,batch_size):
    # def __init__(self, dataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        # self.shuffle = shuffle
        # get the indicies and speaker label
        self.indices = [(i, sample["speaker"]) for i, sample in enumerate(dataset)]
        sample_per_speaker = {}
        for i, speaker in self.indices:
            if speaker not in sample_per_speaker.keys():
                sample_per_speaker[speaker] = 1
            else:
                sample_per_speaker[speaker] += 1
        # if indices are passed, then use only the ones passed (for ddp)
        # if indices is not None:
        #     self.indices = torch.tensor(self.indices)[indices].tolist()
        pooled_indices = []
        # create pool of indices with similar lables
        for i in range(0, len(self.indices), self.batch_size*100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size*100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

    def __iter__(self):
        # if self.shuffle:
        #     random.shuffle(self.indices)
        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]
        # if self.shuffle:
        #     random.shuffle(batches)
        # batch  = [item for subitem in batches for item in subitem]
        for batch in batches:
            for item in batch:
                yield item
        # yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size



class DistributedBatchSamplerSimilarLabel(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size = 10) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BatchSamplerSimilarLabel(self.dataset, batch_size=self.batch_size, indices=indices)
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples//self.batch_size