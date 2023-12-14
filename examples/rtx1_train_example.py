import io
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
from rtx.data_util import describe, preprocess

ds = datasets.load_dataset(
    "jxu124/OpenX-Embodiment",
    "fractal20220817_data", # "bridge",
    split="train",
    cache_dir="datasets_cache",
    streaming=True,  # Comment this out to save dataset to disk.
)  # IterDataset

# describe(next(iter(ds)))

loader = DataLoader(
    ds.map(lambda x: preprocess(x["data.pickle"])), batch_size=1, num_workers=0
)

for batch in loader:
    describe(batch)
    # pprint(batch)
    break

model = RTX1(vit_config=FilmViTConfig(pretrained=True))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

def get_action_vec(action: dict)-> torch.Tensor:
    '''
    action, dict sz: 6
     base_displacement_vector, list(Tensor((1,), torch.float64)) sz: 2
     base_displacement_vertical_rotation, list(Tensor((1,), torch.float64)) sz: 1
     gripper_closedness_action, list(Tensor((1,), torch.float64)) sz: 1
     rotation_delta, list(Tensor((1,), torch.float64)) sz: 3
     terminate_episode, list(Tensor((1,), torch.int64)) sz: 3
     world_vector, list(Tensor((1,), torch.float64)) sz: 3
    '''
    action_vec = torch.zeros(11)
    action_vec[0:1] = torch.concatenate(action['base_displacement_vector']).squeeze()
    action_vec[2] = action['base_displacement_vertical_rotation'][0].squeeze()
    action_vec[3] = action['gripper_closedness_action'][0].squeeze()
    action_vec[4:7] = torch.concatenate(action['rotation_delta']).squeeze()
    action_vec[8] = action['terminate_episode'][0].squeeze()
    action_vec[9:] = torch.concatenate(action['world_vector']).squeeze()
    return action_vec


for epoch in range(10):
    running_loss = 0.0
    for i, batch in tqdm.tqdm(enumerate(loader)):
        for step in batch["steps"]:
            img = step["observation"]["image"]
            text = step["observation"]["natural_language_instruction"]
            actions =get_action_vec(step["observation"]["action"]).unsqueeze(0)

            optimizer.zero_grad()
            outputs = model.train(img, text)
            loss = criterion(outputs, actions)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {running_loss / 10}")
                running_loss = 0.0

# usage
# img = torch.randn(1, 3, 256, 256)
# text = torch.randint(0, 20000, (1, 1024))

# model = RTX2()
# output = model(img, text)
# print(output)
