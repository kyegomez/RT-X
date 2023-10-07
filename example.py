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
