Deep Q Learning for playing Atari Games
=======================================

## Sample results


![Simple DQN playing Atari Enduro-v0](https://github.com/rohitgirdhar/Deep-Q-Networks/raw/master/assets/dqn_enduro.gif) | ![Simple DQN playing Atari Pong-v0](https://github.com/rohitgirdhar/Deep-Q-Networks/raw/master/assets/dqn_pong.gif)


## Training models

### Simple DQN [1]

```bash
python dqn_atari.py \
  --env Enduro-v0 \
  --gpu 0 \
  --model convnet \
  --train_policy epgreedy \
  --std_img \
  --optimizer adam \
  --learning_rate 0.0001
```

### Dueling DQN [4]

Simply replace `--model convnet` with `--model dueling_convnet` in the above command. Also try out other network architectures in `deeprl/networks.py`.

## Performance plots

Following curves compare the
<span style="color:yellow">dueling (yellow)</span>,
<span style="color:green">double (green)</span> and
<span style="color:blue">simple (blue)</span>
deep Q networks.

Episode length

![Episode length](https://github.com/rohitgirdhar/Deep-Q-Networks/raw/master/assets/episode_len.png)

Total Reward over 20 iterations

![Total Reward](https://github.com/rohitgirdhar/Deep-Q-Networks/raw/master/assets/reward.png)

Loss

![Loss](https://github.com/rohitgirdhar/Deep-Q-Networks/raw/master/assets/loss.png)

## References and Acknowledgements

This work was done as a course assignment for the [CMU Deep RL course](https://katefvision.github.io/), so thanks to the instructors for guidance and providing starter code. Also thanks to [Achal](http://www.achaldave.com/) for help in tuning hyperparameters.

[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning.
arXiv preprint arXiv:1312.5602, 2013.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G
Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
Human-level control through deep reinforcement learning. Nature, 518(7540):529–533,
2015.

[3] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with
double q-learning. 2016.

[4] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando
de Freitas. Dueling network architectures for deep reinforcement learning. arXiv preprint
arXiv:1511.06581, 2015.
