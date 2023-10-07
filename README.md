[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# RT-X
Pytorch implementation of the models RT-1-X and RT-2-X from the paper: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"

[Paper Link](https://robotics-transformer-x.github.io/)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install rtx-torch `

# Usage
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

# Architecture

# Todo


# License
MIT

# Citations

