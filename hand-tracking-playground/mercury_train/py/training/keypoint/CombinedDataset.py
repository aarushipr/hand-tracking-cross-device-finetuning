import torch

# from InterHandSequential import AwfulCombinedInterHandDataset
from ArtificialData import ArtificialDataset
from RandoData import RandoDataset

# from settings import datasets_basepath
import local_config
import kpest_header as header

# note everything breaks if artificialdataset is not the biggest

datasets_basepath = local_config.real_datasets_basepath

class AllOfTheDatasetsCombined(torch.utils.data.Dataset):
    def __init__(self):
        amts = []
        datasets = []

        def b(ds, am):
            amts.append(am)
            datasets.append(ds)

        if not header.env_settings.loadfast:
            b(RandoDataset(f"{datasets_basepath}/", "nikitha.csv"), 0.6)
            b(RandoDataset(f"{datasets_basepath}/", "frei_gs.csv"), 0.5)
            b(RandoDataset(f"{datasets_basepath}/", "tom.csv"), 0.8)

            b(RandoDataset(f"{datasets_basepath}/",
                           "panoptic_manual.csv"), 0.8)
            b(RandoDataset(f"{datasets_basepath}/",
                           "panoptic_synth.csv"), 0.8)

        # 0.6+0.5+0.8+0.8+0.8
        # 3.5

        # b(nonechucks.SafeDataset(ArtificialDataset()), 4.0)
        b(ArtificialDataset(), 2.0)

        # b(RandoDataset(f"{datasets_basepath}/",
        #   "panoptic_panoptic.csv"), 0.8)

        biggest_dataset_len = 0
        biggest_dataset_associated_amt = 0
        amt_sum = 0
        for amt, ds in zip(amts, datasets):
            ds_size = len(ds)
            print("sizer", ds_size)
            if ds_size > biggest_dataset_len:
                biggest_dataset_len = ds_size
                biggest_dataset_associated_amt = amt
            amt_sum += amt

        for amt, ds in zip(amts, datasets):
            num_times = int((amt / biggest_dataset_associated_amt)
                            * (biggest_dataset_len / len(ds)))
            ds.num_times_to_repeat = num_times
            # try:
            #   ds.dataset.num_times_to_repeat = num_times
            # except:
            #   print("no")
            print("num", num_times)

        self.ds = torch.utils.data.ConcatDataset(datasets)
        # raise

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


if __name__ == '__main__':

    a = AllOfTheDatasetsCombined()
    raise
