import io
import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

ArrayLike = np.ndarray | list | torch.Tensor


def map_np(input: np.ndarray, idxs: list[int], fn: callable) -> None:
    """Maps a function through a numpy array.

    Args:
        input (np.ndarray): Input.
        fn (callable): Function to map.

    Returns: None
    """
    if sum(input.shape) <= 1:
        fn(input, idxs)
        idxs.pop()
        return

    for i, x in enumerate(input):
        idxs.append(i)
        map_np(x, idxs, fn)


def write_dict_to(name: str, writer: SummaryWriter, input: dict, step: int):
    """Writes a dictionary to tensorboard.

    Args:
        name (str): Name of group to identify values in dict with.
        writer (SummaryWriter): Tensorboard writer.
        input (dict): Input dictionary.
        step (int): Global step value.
    """
    for k, v in input.items():
        v = np.array(v).squeeze()
        if sum(v.shape) <= 1:
            writer.add_scalar(name + "_" + k, v, step)
            continue
        map_np(
            v,
            [],
            lambda x, idxs: writer.add_scalar(
                "{}_{}-{}".format(name, k, "-".join([str(i) for i in idxs])), x, step
            ),
        )


def describe(dic, prefix="", str_built=[]) -> str:
    """Useful to print out the structure of TF Record. ds.info can also be used
        but it does not show lengths of lists and dicts.

    Args:
        dic (dict): Input
        prefix (str, optional): Prefix used for nested indentation. Defaults to ''.
        str_built (str, optional): Desription string built so far. Defaults to ''.
    """
    if not isinstance(dic, dict):
        return ""

    def describe_img(img: bytes):
        img = Image.open(io.BytesIO(img))
        return f"{img.__class__.__name__} sz: { img.size}"

    for k, v in dic.items():
        if isinstance(v, list):
            list_type = ""
            if len(v) > 0:
                v_description = ""
                if isinstance(v[0], torch.Tensor):
                    v_description = f"({tuple(v[0].size())}, {v[0].dtype})"
                elif isinstance(v[0], bytes):
                    v_description = describe_img(v[0])
                list_type = f"({v[0].__class__.__name__ }{v_description})"
            print(f"{prefix} {k}, {v.__class__.__name__}{list_type} sz:" f" {len(v)}")
            if len(v) > 0:
                str_built.append(describe(v[0], prefix + "  "))
        elif isinstance(v, dict):
            print(f"{prefix} {k}, {v.__class__.__name__} sz:" f" {len(v.items())}")
            describe(v, prefix + "  ")
        elif isinstance(v, bytes):
            print(f"{prefix} {k}, {describe_img( v)}")
        elif isinstance(v, str):
            str_built.append(f"{prefix} {k}, {v.__class__.__name__} v: {v}\n")
        else:
            tensor_type = ""
            if isinstance(v, torch.Tensor):
                tensor_type = f"({tuple(v[0].size())}, {v[0].dtype})"
            print(f"{prefix} {k}, {v.__class__.__name__} {tensor_type} ")


def preprocess(dic: any, height=224, width=224):
    """Remove nonetypes from a dict, convert images to numpy arrays and return.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    """
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
    """Resizes images to sz as a numpy array.

    Args:
        dic (dict): Input.

    Returns:
        dict: Output.
    """
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
