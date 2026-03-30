import torch

from InterHandSequential import AwfulCombinedInterHandDataset
from RandoData import RandoDataset

from settings import datasets_basepath


class AllOfTheDatasetsCombined(torch.utils.data.Dataset):
  def __init__(self):
    amts = []
    datasets = []

    def b(ds, am):
      amts.append(am)
      datasets.append(ds)

    b(AwfulCombinedInterHandDataset(), 1.0)

    b(RandoDataset(f"{datasets_basepath}/munge_april26", "nikitha.csv"), 0.6)
    b(RandoDataset(f"{datasets_basepath}/munge_april26", "frei_gs.csv"), 0.5)
    b(RandoDataset(f"{datasets_basepath}/munge_april26", "tom.csv"),     0.8)

    b(RandoDataset(f"{datasets_basepath}/munge_april26",
      "panoptic_manual.csv"), 0.8)
    b(RandoDataset(f"{datasets_basepath}/munge_april26",
      "panoptic_synth.csv"), 0.8)
    b(RandoDataset(f"{datasets_basepath}/munge_april26",
      "panoptic_panoptic.csv"), 0.8)

    b(RandoDataset(f"{datasets_basepath}/munge_april26",
      "artificial_april28_again.csv"), 1.0)

    biggest_dataset_len = 0
    biggest_dataset_associated_amt = 0
    amt_sum = 0
    for amt, ds in zip(amts, datasets):
      ds_size = len(ds)
      if ds_size > biggest_dataset_len:
        biggest_dataset_len = ds_size
        biggest_dataset_associated_amt = amt
      amt_sum += amt

    for amt, ds in zip(amts, datasets):
      num_times = int((amt/biggest_dataset_associated_amt)
                      * (biggest_dataset_len/len(ds)))
      ds.num_times_to_repeat = num_times
      print(num_times)

    self.ds = torch.utils.data.ConcatDataset(datasets)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]


if __name__ == '__main__':

  a = AllOfTheDatasetsCombined()
  raise
