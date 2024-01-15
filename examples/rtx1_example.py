import torch
from rtx.rtx1 import RTX1, FilmViTConfig


def run(pretrained=False):
    """Run RT-X1 example.

    Args:
        pretrained (bool, optional): Whether or not to use a pretrained MaxVit with film (downloads from pytorch).
            Defaults to False.
    """
    model = RTX1(vit_config=FilmViTConfig(pretrained=pretrained))

    video = torch.randn(2, 3, 6, 224, 224)

    instructions = [
        "bring me that apple sitting on the table",
        "please pass the butter",
    ]

    # compute the train logits
    model.train(video, instructions)

    # set the model to evaluation mode
    model.model.eval()

    # compute the eval logits with a conditional scale of 3
    eval_logits = model.run(video, instructions, cond_scale=3.0)
    print(eval_logits.shape)


if __name__ == "__main__":
    run()
