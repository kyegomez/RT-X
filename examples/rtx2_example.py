import torch
from rtx import RTX2


def run():
    # usage
    img = torch.randn(1, 3, 256, 256)
    text = torch.randint(0, 20000, (1, 1024))

    model = RTX2()
    output = model(img, text)
    print(output)


if __name__ == "__main__":
    run()
