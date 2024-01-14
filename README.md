[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# RT-X
Pytorch implementation of the models RT-1-X and RT-2-X from the paper: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models".

Here we implement both model architectures, RTX-1 and RTX-2

[Paper Link](https://robotics-transformer-x.github.io/)

- The RTX-2 Implementation does not natively output for simplicity a 7 dimensional vector but rather text tokens, if you wanted to output 7 dimensional vector you could implement the same token learner as in RTX1


# Appreciation
* Lucidrains
* Agorians

# Install
`pip install rtx-torch `

# Usage
To see detailed usage, run `python run.py --help`.
## RTX1
- RTX1 Usage takes in text and videos
- Does not use Efficient Net yet, we're integrating it now then the implementation will be complete
- Uses SOTA transformer architecture

```python

import torch
from rtx.rtx1 import RTX1, FilmViTConfig

# Use a pre-trained MaxVit model from pytorch
model = RTX1(film_vit_config=FilmViTConfig(pretrained=pretrained))

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


## RTX-2
- RTX-2 takes in images and text and interleaves them to form multi-modal sentences and outputs text tokens not a 7 dimensional vector of x,y,z,roll,pitch,yaw,and gripper
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

## EfficientNetFilm
- Extracts the feature from the given image
```python
from rtx import EfficientNetFilm

model = EfficientNetFilm("efficientnet-b0", 10)

out = model("img.jpeg")


```
# Model Differences from the Paper Implementation
## RT-1
The main difference here is the substitution of a Film-EfficientNet backbone (pre-trained EfficientNet-B3 with Film layers inserted) with a MaxViT model.



# Tests
I created a single tests file that uses pytest to run tests on all the modules, RTX1, RTX2, EfficientNetFil, first git clone and get into the repository, install the requirements.txt with pip then run this:

`python -m pytest tests/tests.py`

# License
MIT

# Citations
```bibtex
@misc{open_x_embodiment_rt_x_2023,
title={Open {X-E}mbodiment: Robotic Learning Datasets and {RT-X} Models},
author = {Open X-Embodiment Collaboration and Abhishek Padalkar and Acorn Pooley and Ajinkya Jain and Alex Bewley and Alex Herzog and Alex Irpan and Alexander Khazatsky and Anant Rai and Anikait Singh and Anthony Brohan and Antonin Raffin and Ayzaan Wahid and Ben Burgess-Limerick and Beomjoon Kim and Bernhard Schölkopf and Brian Ichter and Cewu Lu and Charles Xu and Chelsea Finn and Chenfeng Xu and Cheng Chi and Chenguang Huang and Christine Chan and Chuer Pan and Chuyuan Fu and Coline Devin and Danny Driess and Deepak Pathak and Dhruv Shah and Dieter Büchler and Dmitry Kalashnikov and Dorsa Sadigh and Edward Johns and Federico Ceola and Fei Xia and Freek Stulp and Gaoyue Zhou and Gaurav S. Sukhatme and Gautam Salhotra and Ge Yan and Giulio Schiavi and Hao Su and Hao-Shu Fang and Haochen Shi and Heni Ben Amor and Henrik I Christensen and Hiroki Furuta and Homer Walke and Hongjie Fang and Igor Mordatch and Ilija Radosavovic and Isabel Leal and Jacky Liang and Jaehyung Kim and Jan Schneider and Jasmine Hsu and Jeannette Bohg and Jeffrey Bingham and Jiajun Wu and Jialin Wu and Jianlan Luo and Jiayuan Gu and Jie Tan and Jihoon Oh and Jitendra Malik and Jonathan Tompson and Jonathan Yang and Joseph J. Lim and João Silvério and Junhyek Han and Kanishka Rao and Karl Pertsch and Karol Hausman and Keegan Go and Keerthana Gopalakrishnan and Ken Goldberg and Kendra Byrne and Kenneth Oslund and Kento Kawaharazuka and Kevin Zhang and Keyvan Majd and Krishan Rana and Krishnan Srinivasan and Lawrence Yunliang Chen and Lerrel Pinto and Liam Tan and Lionel Ott and Lisa Lee and Masayoshi Tomizuka and Maximilian Du and Michael Ahn and Mingtong Zhang and Mingyu Ding and Mohan Kumar Srirama and Mohit Sharma and Moo Jin Kim and Naoaki Kanazawa and Nicklas Hansen and Nicolas Heess and Nikhil J Joshi and Niko Suenderhauf and Norman Di Palo and Nur Muhammad Mahi Shafiullah and Oier Mees and Oliver Kroemer and Pannag R Sanketi and Paul Wohlhart and Peng Xu and Pierre Sermanet and Priya Sundaresan and Quan Vuong and Rafael Rafailov and Ran Tian and Ria Doshi and Roberto Martín-Martín and Russell Mendonca and Rutav Shah and Ryan Hoque and Ryan Julian and Samuel Bustamante and Sean Kirmani and Sergey Levine and Sherry Moore and Shikhar Bahl and Shivin Dass and Shuran Song and Sichun Xu and Siddhant Haldar and Simeon Adebola and Simon Guist and Soroush Nasiriany and Stefan Schaal and Stefan Welker and Stephen Tian and Sudeep Dasari and Suneel Belkhale and Takayuki Osa and Tatsuya Harada and Tatsuya Matsushima and Ted Xiao and Tianhe Yu and Tianli Ding and Todor Davchev and Tony Z. Zhao and Travis Armstrong and Trevor Darrell and Vidhi Jain and Vincent Vanhoucke and Wei Zhan and Wenxuan Zhou and Wolfram Burgard and Xi Chen and Xiaolong Wang and Xinghao Zhu and Xuanlin Li and Yao Lu and Yevgen Chebotar and Yifan Zhou and Yifeng Zhu and Ying Xu and Yixuan Wang and Yonatan Bisk and Yoonyoung Cho and Youngwoon Lee and Yuchen Cui and Yueh-hua Wu and Yujin Tang and Yuke Zhu and Yunzhu Li and Yusuke Iwasawa and Yutaka Matsuo and Zhuo Xu and Zichen Jeff Cui},
howpublished  = {\url{https://arxiv.org/abs/2310.08864}},
year = {2023},
}
```

# Todo
- Integrate EfficientNetFilm with RTX-1
- Create training script for RTX-1 by unrolling observations and do basic cross entropy in first rt-1
- Use RTX-2 dataset on huggingface
- [Check out the project board for more tasks](https://github.com/users/kyegomez/projects/10/views/1)