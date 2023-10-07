[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# RT-X
Pytorch implementation of the models RT-1-X and RT-2-X from the paper: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"

Here we implement both model architectures, RTX-1 and RTX-2

[Paper Link](https://robotics-transformer-x.github.io/)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install rtx-torch `

# Usage
- RTX1 Usage takes in text and videos

```python

import torch
from rtx.rtx1 import RTX1

model = RTX1()

video = torch.randn(2, 3, 6, 224, 224)

instructions = ["bring me that apple sitting on the table", "please pass the butter"]

# compute the train logits
train_logits = model.train(video, instructions)

# set the model to evaluation mode
model.model.eval()

# compute the eval logits with a conditional scale of 3
eval_logits = model.run(video, instructions, cond_scale=3.0)
print(eval_logits.shape)
```

- RTX-2 takes in images and text and interleaves them to form multi-modal sentences:
```python

import torch
from rtx import RTX2

# usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = RTX2()
output = model(img, text)
print(output)


```

# License
MIT

# Citations


# Todo
- Integrate Efficient net with RT-1 and RT-2
- create training script for both models
- Provide a table of all the datasets