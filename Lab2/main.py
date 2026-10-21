from copy_files import copy_files
from build_annotation import build_annotation
from shuffle_dataset import shuffle_dataset
from dataset_iterator import DatasetIterator

# copy_files('../Lab1/dataset/zebra', './dataset')
# copy_files('../Lab1/dataset/bay horse', './dataset')
# build_annotation()
# shuffle_dataset()


bay_horse_dataset = DatasetIterator("bay horse")
for path in bay_horse_dataset:
    print(path)

zebra_dataset = DatasetIterator("zebra")
print(next(zebra_dataset))
print(next(zebra_dataset))
