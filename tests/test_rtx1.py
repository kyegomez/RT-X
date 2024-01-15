import unittest
import torch
from rtx.rtx1 import RTX1, FilmViTConfig, RT1Config


class RTX1Test(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_frames = 6
        self.num_actions = 11
        self.num_action_bins = 256

        self.video = torch.randn(self.batch_size, 3, self.num_frames, 224, 224)
        self.instructions = [
            "bring me that apple sitting on the table",
            "please pass the butter",
        ]

        rt1_config = RT1Config(
            num_actions=self.num_actions,
            action_bins=self.num_action_bins,
        )
        self.rtx1 = RTX1(rt1_config)
        self.rtx1_pretrained = RTX1(rt1_config, FilmViTConfig(pretrained=True))
        self.expected_logits_shape = torch.Size(
            [
                self.batch_size,
                self.num_frames,
                self.num_actions,
                self.num_action_bins,
            ]
        )

    def test_default_pretrained_has_same_shape(self):
        # Tests the general shape as the pretrained version from pytorch has
        # different layernorm and conv2dnorm implementations.

        assert len(self.rtx1.vit.layers) == len(self.rtx1_pretrained.vit.layers)

    def test_default_train_eval(self):
        train_logits = self.rtx1.train(self.video, self.instructions)

        assert train_logits.shape == self.expected_logits_shape
        self.rtx1.model.eval()

        # compute the eval logits with a conditional scale of 3
        eval_logits = self.rtx1.run(self.video, self.instructions, cond_scale=3.0)
        assert eval_logits.shape == self.expected_logits_shape

    def test_pretrained_train_eval(self):
        train_logits = self.rtx1_pretrained.train(self.video, self.instructions)

        assert train_logits.shape == self.expected_logits_shape
        self.rtx1.model.eval()

        # compute the eval logits with a conditional scale of 3
        eval_logits = self.rtx1_pretrained.run(
            self.video, self.instructions, cond_scale=3.0
        )
        assert eval_logits.shape == self.expected_logits_shape


if __name__ == "__main__":
    unittest.main()
