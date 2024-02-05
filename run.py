from examples import rtx1_example, train_example
from rtx import RTX1, RTX2
from rtx.rtx1 import FilmViTConfig
from absl import app, flags, logging

from . import rtx2_example

REGISTRY = {
    "rtx1": RTX1,
    "rtx2": RTX2,
}

MODES = ["inference", "train"]

EXAMPLE_SCRIPTS = {
    "rtx1": rtx1_example,
    "rtx2": rtx2_example,
}

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "pretrained_vit", False, "Whether to use a  pretrained ViT as a backbone or not."
)
flags.DEFINE_enum("model", "rtx1", REGISTRY.keys(), "Model to choose from.")
flags.DEFINE_enum("mode", "inference", MODES, "Experiment mode to run.")


def main(_):
    if FLAGS.mode == "inference":
        EXAMPLE_SCRIPTS[FLAGS.model].run()
    elif FLAGS.mode == "train":
        if FLAGS.pretrained_vit and FLAGS.model == "rtx2":
            logging.fatal(
                "Option `pretrained_vit` is not available for model {} ".format(
                    FLAGS.model
                )
            )
        model = REGISTRY[FLAGS.model](
            vit_config=FilmViTConfig(pretrained=FLAGS.pretrained_vit)
        )
        train_example.run(model)


if __name__ == "__main__":
    app.run(main)
