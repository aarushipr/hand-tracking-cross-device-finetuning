import os
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

        # num_times_to_repeat scales every dataset proportionally, whichever is largest.

        # Skip a real dataset whose CSV isn't munged yet rather than blocking all training.
        def b_if_present(csv_name, weight):
            csv_path = os.path.join(f"{datasets_basepath}/", csv_name)
            if os.path.exists(csv_path):
                b(RandoDataset(f"{datasets_basepath}/", csv_name), weight)
            else:
                print(f"[CombinedDataset] Skipping {csv_name} — not found at {csv_path}")

        # freihand and tom are held out for val/test; adding them here contaminates evaluation.
        if not header.env_settings.loadfast:
            b_if_present("nikitha.csv", 0.6)
            b_if_present("panoptic_manual.csv", 0.8)
            b_if_present("panoptic_synth.csv", 0.8)
            # Landmark permutation confirmed 2026-07-21 against real skeleton data, not guessed.
            # Egocentric XR headset data, the most device-relevant real source available.
            b_if_present("umetrack.csv", 0.8)

        b(ArtificialDataset(), 2.0)

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
