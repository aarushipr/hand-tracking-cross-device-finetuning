import os
import torch

from py.training.detection.HMDHandRectsDataset import HMDHandRectsDataset
from py.training.detection.EpicKitchensDataset import EpicKitchensDataset
from py.training.detection.DarknetDataset import DarknetDataset
from py.training.detection.SyntheticDetectionDataset import SyntheticDetectionDataset
import py.training.detection.local_config as local_config


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, wrap_dataset, num_times):
        self.wrap_dataset = wrap_dataset

        self.num_times = num_times
        self.actual_size = len(self.wrap_dataset)

    def __len__(self):
        return self.actual_size*self.num_times

    def __getitem__(self, idx):
        return self.wrap_dataset[idx % self.actual_size]


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self):
        amts = []
        datasets = []

        def b(ds, am):
            amts.append(am)
            datasets.append(ds)

        # Primary source: the synthetic generator KeyNet trains on, boxes from the 3D joints.
        synthetic_path = getattr(local_config, "artificial_dataset_path", None)
        if synthetic_path and os.path.isdir(synthetic_path):
            # Highest weight: the primary source, matching KeyNet's synthetic-first design.
            b(SyntheticDetectionDataset(), 4)
        else:
            print("[CombinedDataset] Skipping synthetic detection data — "
                  "local_config.artificial_dataset_path not set or missing. "
                  "DetNet will have no training signal unless a real source "
                  "below is available.")

        # Real sources are val/test-oriented now; each is skipped individually if missing.
        hmdhandrect_datasets = []
        hmdhandrect_root = local_config.hmdhandrects_location
        # subject02 is the validation set; adding it here would contaminate evaluation.
        for seq_name in [
            "train_subject00_sequence00",
            "train_subject00_sequence01",
            "train_subject00_sequence02",
            "train_subject00_sequence03",
            "train_subject01_sequence00",
            "train_subject01_sequence01",
        ]:
            seq_root = f"{hmdhandrect_root}/sequences/{seq_name}"
            ann_path = os.path.join(seq_root, HMDHandRectsDataset.ann)
            if os.path.exists(ann_path):
                hmdhandrect_datasets.append(HMDHandRectsDataset(seq_root))
            else:
                print(f"[CombinedDataset] Skipping HMDHandRects {seq_name} — not found at {ann_path}")

        if hmdhandrect_datasets:
            # HMDHandRects: egocentric XR; the most device-relevant data, weight up.
            b(torch.utils.data.ConcatDataset(hmdhandrect_datasets), 3)
        else:
            print("[CombinedDataset] Skipping HMDHandRects entirely — no sequences found")

        egohands_labels_dir = os.path.join(local_config.egohands_convert, "labels", "train")
        if os.path.isdir(egohands_labels_dir):
            # EgoHands: egocentric but not XR hardware, still useful, moderate weight.
            b(DarknetDataset(local_config.egohands_convert), 1)
        else:
            print(f"[CombinedDataset] Skipping EgoHands — not found at {egohands_labels_dir}")

        if os.path.isdir(local_config.kitchens_annotations) and os.path.isdir(local_config.kitchens_images):
            # EpicKitchens: GoPro chest-mounted, wrong device type; kept for diversity, low weight.
            b(EpicKitchensDataset(), 2)
        else:
            print(f"[CombinedDataset] Skipping EpicKitchens — not found at "
                  f"{local_config.kitchens_annotations} / {local_config.kitchens_images}")

        if not datasets:
            raise RuntimeError(
                "[CombinedDataset] No detection training sources available at all — "
                "the synthetic source (local_config.artificial_dataset_path) and all "
                "real sources (HMDHandRects, EgoHands, EpicKitchens) are missing/"
                "unconfigured in local_config.py. Set artificial_dataset_path to the "
                "same directory keypoint's local_config.py uses — that alone is "
                "enough to train."
            )

        repeat_datasets = []

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
            num_times = int((amt/biggest_dataset_associated_amt)
                            * (biggest_dataset_len/len(ds)))
            # ds.num_times_to_repeat = num_times
            repeat_datasets.append(RepeatDataset(ds, num_times))
            # try:
            #   ds.dataset.num_times_to_repeat = num_times
            # except:
            #   print("no")
            print("num", num_times)

        self.ds = torch.utils.data.ConcatDataset(repeat_datasets)
        # raise

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


if __name__ == '__main__':

    a = CombinedDataset()
    raise
