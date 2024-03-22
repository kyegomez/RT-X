import argparse
from examples import (
    rtx1_example,
    rtx1_pretrained_example,
    rtx2_example,
)

EXAMPLES = {
    "rtx1": rtx1_example,
    "rtx1-pretrained": rtx1_pretrained_example,
    "rtx2": rtx2_example,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="rtx1",
        choices=EXAMPLES.keys(),
        help="Example to choose from",
    )
    args = parser.parse_args()
    EXAMPLES[args.model_type].run()
