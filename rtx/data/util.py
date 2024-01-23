import io
import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from absl import logging
import tqdm

def map_np(input: np.ndarray, idxs: list[int], fn: callable) -> None:
    '''Maps a function through a numpy array.

    Args:
        input (np.ndarray): Input.
        fn (callable): Function to map.

    Returns: None
    '''
    if sum(input.shape) <= 1:
        fn(input, idxs)
        idxs.pop()
        return

    for i, x in enumerate(input):
        idxs.append(i)
        map_np(x, idxs, fn)


def write_dict_to(name: str, writer: SummaryWriter, input: dict, step: int):
    '''Writes a dictionary to tensorboard.

    Args:
        name (str): Name of group to identify values in dict with.
        writer (SummaryWriter): Tensorboard writer.
        input (dict): Input dictionary.
        step (int): Global step value.
    '''
    for k, v in input.items():
        v = np.array(v).squeeze()
        if sum(v.shape) <= 1:
            writer.add_scalar(name + '_' + k, v, step)
            continue
        map_np(
            v, [], lambda x, idxs: writer.add_scalar(
                '{}_{}-{}'.format(name, k, '-'.join([str(i)
                                                     for i in idxs])), x, step))


def describe(dic, prefix='') -> None:
    '''Useful to print out the structure of TF Record. ds.info can also be used
        but it does not show lengths of lists and dicts.

    Args:
        dic (dict): Input
        prefix (str, optional): Prefix used for nested indentation. Defaults to ''.
        str_built (str, optional): Desription string built so far. Defaults to ''.
    '''
    if not isinstance(dic, dict):
        return

    def describe_img(img: bytes):
        img = Image.open(io.BytesIO(img))
        return f'{img.__class__.__name__} sz: { img.size}'

    for k, v in dic.items():
        if isinstance(v, list):
            list_type = ''
            if len(v) > 0:
                v_description = ''
                if isinstance(v[0], torch.Tensor):
                    v_description = (
                        f"({tuple(v[0].size())}, {v[0].dtype})"
                    )
                elif isinstance(v[0], bytes):
                    v_description = describe_img(v[0])
                list_type = (
                    f"({v[0].__class__.__name__ }{v_description})"
                )
            print(
                f"{prefix} {k}, {v.__class__.__name__}{list_type} sz:"
                f" {len(v)}"
            )
            if len(v) > 0:
             describe(v[0], prefix + '  ')
        elif isinstance(v, dict):
            print(
                f"{prefix} {k}, {v.__class__.__name__} sz:"
                f" {len(v.items())}"
            )
            describe(v, prefix + "  ")
        elif isinstance(v, bytes):
            print(f"{prefix} {k}, {describe_img( v)}")
        elif isinstance(v, str):
            print(f'{prefix} {k}, {v.__class__.__name__} v: {v}')
        else:
            tensor_type = ''
            if isinstance(v, torch.Tensor):
                tensor_type = f"({tuple(v[0].size())}, {v[0].dtype})"
            print(
                f"{prefix} {k}, {v.__class__.__name__} {tensor_type} "
            )


def preprocess(dic: any, height=224, width=224):
    '''Remove nonetypes from a dict, convert images to numpy arrays and return.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    '''
    if isinstance(dic, bytes):
        img = Image.open(io.BytesIO(dic))
        return np.array(img.resize((width, height)))

    if not isinstance(dic, dict):
        return dic

    to_remove = []
    for k, v in dic.items():
        if isinstance(v, list):
            processed = []
            for vv in v:
                processed.append(preprocess(vv, height, width))
            dic[k] = processed
        elif v is None:
            to_remove.append(k)
        else:
            dic[k] = preprocess(v, height, width)
    for k in to_remove:
        del dic[k]
    return dic



def format_imgs(dic: any, sz: int):
    '''Resizes images to sz as a numpy array.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    '''
    if isinstance(dic, bytes):
        img = Image.open(io.BytesIO(dic))
        return np.array(img.resize((sz, sz)))
        return np.array(img.resize((sz, sz)))

    if not isinstance(dic, dict):
        return dic

    for k, v in dic.items():
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = format_imgs(v, sz)
        else:
            dic[k] = format_imgs(v, sz)
    return dic



def compare_model_to_dataset(
        model: torch.nn.Module,
        ds: Dataset,
        max_steps: int = 114,
):
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
 

    loader = DataLoader(ds.map(lambda x: preprocess(x["data.pickle"], height=256, width=320)),
                        batch_size=1,
                        num_workers=0)
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
       
 
            rt1_act = model(text, img.numpy().squeeze(), reward=step["reward"].item())

            rt1_act_world_vector_sum += rt1_act['world_vector'].squeeze()
            rt1_act_rotation_delta_sum += rt1_act['rotation_delta'].squeeze()
            rt1_act['world_vector_sum'] = rt1_act_world_vector_sum
            rt1_act['rotation_delta_sum'] = rt1_act_rotation_delta_sum
            write_dict_to("rt1_act", writer, rt1_act, step_count)
            writer.add_image("image", img, step_count, dataformats="HWC")


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

            reward = step["reward"].squeeze()
            is_first = step["is_first"].squeeze()
            is_last = step["is_last"].squeeze()
            is_terminal = step["is_terminal"].squeeze()

            writer.add_scalar("reward", reward, step_count)
            writer.add_scalar("is_first", is_first, step_count)
            writer.add_scalar("is_last", is_last, step_count)
            writer.add_scalar("is_terminal", is_terminal, step_count)
            writer.flush()
            if step_count > max_steps:
                break
            step_count += 1
        world_vector_sum = np.zeros(3)
        rotation_delta_sum = np.zeros(3)
        rt1_act_world_vector_sum = np.zeros(3)
        rt1_act_rotation_delta_sum = np.zeros(3)

