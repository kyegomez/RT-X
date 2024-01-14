from absl import logging, flags
import torch
from torch import nn, optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from rtx import RTX1
from rtx.rtx1 import FilmViTConfig
import datasets
import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from rtx.data_util import describe, preprocess, write_dict_to
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from orca.data.dataset import make_interleaved_dataset
from orca.data.oxe import make_oxe_dataset_kwargs_and_weights
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import tqdm

DATA_PATH = "gs://rail-orca-central2/resize_256_256"

tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS


# from ray.rllib.algorithms.bc import BCConfig



def tokenize_action(actions: list[dict]) -> torch.Tensor:
    '''
    action, dict sz: 6
     base_displacement_vector, list(Tensor((1,), torch.float64)) sz: 2
     base_displacement_vertical_rotation, list(Tensor((1,), torch.float64)) sz: 1
     gripper_closedness_action, list(Tensor((1,), torch.float64)) sz: 1
     rotation_delta, list(Tensor((1,), torch.float64)) sz: 3
     terminate_episode, list(Tensor((1,), torch.int64)) sz: 3
     world_vector, list(Tensor((1,), torch.float64)) sz: 3
    '''

    # for k, v in action.items():
    #     action[k] = torch.tensor(v)
    # action_vec = torch.zeros(11)
    # action_vec[0:1] = torch.concatenate(action['base_displacement_vector']).squeeze()
    # action_vec[2] = action['base_displacement_vertical_rotation'][0].squeeze()
    # action_vec[3] = action['gripper_closedness_action'][0].squeeze()
    # action_vec[4:7] = torch.concatenate(action['rotation_delta']).squeeze()
    # action_vec[8] = action['terminate_episode'][0].squeeze()
    # action_vec[9:] = torch.concatenate(action['world_vector']).squeeze()
    # return action_vec
    pass




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
            yield sample

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


dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
    "oxe_magic_soup",
    DATA_PATH,
    load_camera_views=("primary"),
)

dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    shuffle_buffer_size=1000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
    batch_size=None,  # batching will be handles in PyTorch Dataloader object
    balance_weights=True,
    traj_transform_kwargs=dict(
        goal_relabeling_strategy="uniform",
        window_size=2,
        future_action_window_size=3,
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
            "wrist": dict(
                random_brightness=[0.1],
                random_contrast=[0.9, 1.1],
                random_saturation=[0.9, 1.1],
                random_hue=[0.05],
                augment_order=[
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
        },
        resize_size=dict(
            primary=(256, 256),
            wrist=(128, 128),
        ),
        num_parallel_calls=200,
    ),
    traj_transform_threads=48,
    traj_read_threads=48,
)


pytorch_dataset = TorchRLDSDataset(dataset)
dataloader = DataLoader(
    pytorch_dataset,
    batch_size=16,
    num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
)

for i, sample in tqdm.tqdm(enumerate(dataloader)):
    if i == 5000:
        break


for epoch in range(10):
    running_loss = 0.0
    # for i, batch in tqdm.tqdm(enumerate(loader)):
    #     for step in batch["steps"]:
    #         img = step["observation"]["image"]
    #         text = step["observation"]["natural_language_instruction"]
    #         logging.debug(step["observation"].keys())
    #         actions =get_action_vec(step["observation"]["action"]).squeeze()

    #         optimizer.zero_grad()
    #         outputs = model.train(img, text)
    #         loss = criterion(outputs, actions)
    #         loss.backward()

    #         optimizer.step()
    #         running_loss += loss.item()
    #         if i % 10 == 9:
    #             logging.debug(f"Epoch: {epoch}, Batch: {i}, Loss: {running_loss / 10}")
    #             running_loss = 0.0


def run(
        model: torch.nn.Module,
        max_steps: int = 114,
        # config: BCConfig,
        dataset_path: str = "jxu124/OpenX-Embodiment",
        dataset_name: str = "fractal20220817_data",
        split: str = "train",
        streaming: bool = True):
    '''Run training and eval for Behavior Cloning.

    Args:
        model (torch.nn.Module): Model to train.
        config (BCConfig): Behavior Cloning config.
        dataset_path (str): Huggingface dataset path.
        dataset_name (str): Huggingface dataset name.
        split (str): Train or eval split.
        streaming (bool, optional): Whether to stream or download the dataset.
          If true, an IterableDatset from the transformers library will be used.
          Defaults to True.
    '''
    logging.info("Fetching dataset {}/{}".format(dataset_path, dataset_name))
    ds = datasets.load_dataset(dataset_path,
                               dataset_name,
                               streaming=streaming,
                               split=split,
                               cache_dir="dataset_cache")

    loader = DataLoader(ds.map(lambda x: preprocess(x["data.pickle"], height=256, width=320)),
                        batch_size=1,
                        num_workers=0)
    inference = InferenceServer(model_key=FLAGS.model_key)
    writer = SummaryWriter()

    logging.debug(ds.info)
    step_count = 0
    world_vector_sum = np.zeros(3)
    rotation_delta_sum = np.zeros(3)
    rt1_act_world_vector_sum = np.zeros(3)
    rt1_act_rotation_delta_sum = np.zeros(3)
    seen_instructions = set()
    for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
        for step in batch["steps"]:
            obs = step["observation"]
            img = obs["image"]['bytes'].squeeze()
            text = obs["natural_language_instruction"][0]

            if step_count == 0:
                logging.debug(describe(step))
            if text not in seen_instructions and logging.level_debug():
                seen_instructions.add(text)
                print("Step {} ################### Batch {} Instruction: {}".format(step_count, batch_idx, text))
       
 
            rt1_act = inference(text, img.numpy().squeeze(), reward=step["reward"].item())
            # rt1_policy_state = inference.get_policy_state(text, img.squeeze())    `
   
            # rt1_policy_state = {k: np.array(v).squeeze(0).squeeze() for k, v in rt1_policy_state.items() if k != 'image'}

            rt1_act_world_vector_sum += rt1_act['world_vector'].squeeze()
            rt1_act_rotation_delta_sum += rt1_act['rotation_delta'].squeeze()
            rt1_act['world_vector_sum'] = rt1_act_world_vector_sum
            rt1_act['rotation_delta_sum'] = rt1_act_rotation_delta_sum
            write_dict_to("rt1_act", writer, rt1_act, step_count)
            writer.add_image("image", img, step_count, dataformats="HWC")
            # write_dict_to("rt1_policy_state", writer, rt1_policy_state, step_count)
            # rt1_image = rt1_policy_state['image'].squeeze()
            # writer.add_image("rt1_image0",rt1_image[0], step_count, dataformats="HWC")
            # writer.add_image("rt1_image1",rt1_image[1], step_count, dataformats="HWC")
            # writer.add_image("rt1_image2",rt1_image[2], step_count, dataformats="HWC")
            # writer.add_image("rt1_image3",rt1_image[3], step_count, dataformats="HWC")
            # writer.add_image("rt1_image4",rt1_image[4], step_count, dataformats="HWC")
            # writer.add_image("rt1_image5",rt1_image[5], step_count, dataformats="HWC")
            # del rt1_policy_state['image']
            # for i, act_token in enumerate(rt1_policy_state['action_tokens']):
            #     for j, token in enumerate(act_token):
            #         writer.add_scalar("policy_rt1_act_token-{0}-{1}".format(i, j), token, step_count)
            # writer.add_scalar("policy_rt1_t", rt1_policy_state['t'], step_count)
            # writer.add_scalar("policy_rt1_step", rt1_policy_state['step_num'], step_count)

            # obs["orientation_box"] = torch.tensor(obs["orientation_box"])
            # obs["robot_orientation_positions_box"] = torch.tensor(
            #     obs["robot_orientation_positions_box"])
            # obs["workspace_bounds"] = torch.tensor(obs["workspace_bounds"])

            if step_count == 0:
                writer.add_embedding(obs["natural_language_embedding"], global_step=step_count)
                writer.add_text("natural_language_instruction", text, global_step=step_count)
            del obs["image"]
            del obs["natural_language_embedding"]
            del obs["natural_language_instruction"]
            del obs["orientation_box"] # NaaN
            write_dict_to("obs", writer, obs, step_count)
            world_vector_sum += np.array(step['action']['world_vector']).squeeze()
            rotation_delta_sum += np.array(step['action']['rotation_delta']).squeeze()
            step['action']['world_vector_sum'] = world_vector_sum
            step['action']['rotation_delta_sum'] = rotation_delta_sum
            write_dict_to("action", writer, step["action"], step_count)            

            # obs = {k: torch.cat(v) for k, v in obs.items()}
            # action = {k: torch.cat(v) for k, v in step["action"].items()}
            # action= torch.tensor(step["action"], dtype=torch.float16).squeeze()
            reward = step["reward"].squeeze()
            is_first = step["is_first"].squeeze()
            is_last = step["is_last"].squeeze()
            is_terminal = step["is_terminal"].squeeze()
            # # actions =get_action_vec(step["observation"]["action"]).squeeze()
            # writer.add_scalars("observation", obs, step_count)
            # writer.add_scalars("action", action, step_count)
            writer.add_scalar("reward", reward, step_count)
            writer.add_scalar("is_first", is_first, step_count)
            writer.add_scalar("is_last", is_last, step_count)
            writer.add_scalar("is_terminal", is_terminal, step_count)
            writer.flush()
            if step_count > max_steps:
                # break
                pass
            step_count += 1
        world_vector_sum = np.zeros(3)
        rotation_delta_sum = np.zeros(3)
        rt1_act_world_vector_sum = np.zeros(3)
        rt1_act_rotation_delta_sum = np.zeros(3)
        # break
