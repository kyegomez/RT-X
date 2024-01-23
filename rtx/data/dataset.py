from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights, make_oxe_dataset_kwargs
from octo.data.utils.data_utils import NormalizationType
from dlimp import DLataset
import torch
import numpy as np
import datasets
from absl import logging
from rtx.data.registry import DATASET_MIXES



class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield {'observation': {"image_primary": sample['observation']['image_primary']},
                                #    "image_wrist": sample['observation']['image_wrist']},
                   'action': sample['action'],
                   'language_instruction': sample['task']['language_instruction'].decode()}

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)



def get_interleaved_oxe_dataset(mix_name: str = "eef_pose_magic_soup", data_dir: str = "gs://gresearch/robotics") -> DLataset:

    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        mix_name,
        "gs://gresearch/robotics",
        load_camera_views=("primary", "wrist"),
       action_proprio_normalization_type= NormalizationType.NONE,
    )
    logging.info("Creating interleaved OXE dataset {} from {}".format(mix_name, data_dir))
    return make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        shuffle_buffer_size=500000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            window_size=6,
            future_action_window_size=0,
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(224, 224),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

def get_single_oxe_dataset(name: str = "fractal20220817_data", data_dir: str = "gs://gresearch/robotics") -> DLataset:
    dataset_kwargs = make_oxe_dataset_kwargs(
    # see octo/data/oxe/oxe_dataset_configs.py for available datasets
    # (this is a very small one for faster loading)
    # "austin_buds_dataset_converted_externally_to_rlds",
    name,
    # can be local or on cloud storage (anything supported by TFDS)
    # "/path/to/base/oxe/directory",
    "gs://gresearch/robotics",
    action_proprio_normalization_type= NormalizationType.NONE,
    )
    logging.info("Creating single OXE dataset {} from {}".format(name, data_dir))
    return make_single_dataset(dataset_kwargs, train=True,
      traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            window_size=6,
            future_action_window_size=0,
            subsample_length=100,
        ),
           frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(224, 224),
            ),
            num_parallel_calls=30,
        ),).flatten().shuffle(buffer_size=100)

def get_oxe_dataset(name: str = "fractal20220817_data") -> DLataset:
    if name in DATASET_MIXES:
        return get_interleaved_oxe_dataset(name)
    else:
        return get_single_oxe_dataset(name)

def get_hf_dataset(  
        dataset_path: str = "jxu124/OpenX-Embodiment",
        dataset_name: str = "fractal20220817_data",
        split: str = "train",
        streaming: bool = True):
    logging.info("Fetching dataset {}/{}".format(dataset_path, dataset_name))
    ds = datasets.load_dataset(dataset_path,
                               dataset_name,
                               streaming=streaming,
                               split=split,
                               cache_dir="dataset_cache")
    return ds