from absl import flags
import torch
from torch import nn, optim
import tqdm
from torch.utils.data import DataLoader
from rtx.data.util import write_dict_to
from rtx.data.dataset import get_oxe_dataset, TorchRLDSDataset
from tensorboardX import SummaryWriter
import os
# tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")




def run(model: torch.nn.Module, action_tokenizer):
    writer = SummaryWriter()
    dataset = get_oxe_dataset(FLAGS.dataset_name)
 
    pytorch_dataset = TorchRLDSDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    step_num = 0
    for epoch in range(FLAGS.num_epochs):
        print(f'epoch {epoch}')
        for i, sample in tqdm.tqdm(enumerate(dataloader)):
            # batch, frames, height, width, channels -> batch, channels, frames, height, width
            video = torch.permute(sample['observation']['image_primary'],(0,4,1,2,3)) / 255.0
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action']).reshape(-1,1).squeeze().long()

            optimizer.zero_grad()
            out = model.train(video, instructions).reshape(-1, 256)
            loss = criterion(out, ground_truth)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, step_num)
            
            if (i+1) % 10 == 0:
                # writer.add_image('last img first sample', sample['observation']['image_primary'][0,-1,:,:,:].numpy(), step_num)
                write_dict_to('last_action_first_batch_sample', writer, {'act': sample['action'][0,-1,:]} , step_num)
                writer.add_text('instruction_first_batch_sample', sample['language_instruction'][0], step_num)
            step_num += 1

        # save model
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{FLAGS.model}_{FLAGS.dataset_name}_step{step_num}.pt')
