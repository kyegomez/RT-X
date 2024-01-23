from examples import rtx1_example, rtx2_example, train_example
from rtx import RTX1, RTX2
from rtx.rtx1 import FilmViTConfig
from rtx.action_tokenization import RTX1ActionTokenizer
from absl import app, flags, logging

REGISTRY = {
    "rtx1": {
        "model": RTX1,
        "action_tokenizer": RTX1ActionTokenizer,
    },
    "rtx2": {
        "model": RTX2,
        "action_tokenizer": None,
    },
}


MODES = ["inference", "train"]

EXAMPLE_SCRIPTS = {
    "rtx1": rtx1_example,
    "rtx2": rtx2_example,
}

FLAGS = flags.FLAGS
flags.DEFINE_boolean("pretrained_vit", False, "Whether to use a  pretrained ViT as a backbone or not.")
flags.DEFINE_enum("model", "rtx1", REGISTRY.keys(), "Model to choose from.")
flags.DEFINE_enum("mode", "inference", MODES, "Experiment mode to run.")

def main(_):
    if FLAGS.mode == "inference":
        EXAMPLE_SCRIPTS[FLAGS.model].run()
    elif FLAGS.mode == "train":
        if FLAGS.pretrained_vit and FLAGS.model == "rtx2":
            logging.fatal("Option `pretrained_vit` is not available for model {} ".format(FLAGS.model))
        model = REGISTRY[FLAGS.model]['model'](vit_config= FilmViTConfig(pretrained= FLAGS.pretrained_vit))
        action_tokenizer = REGISTRY[FLAGS.model]['action_tokenizer']()
        train_example.run(model, action_tokenizer)

if __name__ == "__main__":
    app.run(main)
