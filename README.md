commands to run:
source .venv/bin/activate

pkill -f tensorboard || true
tensorboard --logdir runs/tb


**Results**
Baseline (bipedalwalker_baseline_V1):
Evaluation over 10 episodes: mean=202.46 +- 146.54
Evaluation over 20 episodes: mean=284.15 +- 55.43
Potential (bipedalwalker_potential_v):
Evaluation over 10 episodes: mean=296.78 +- 2.69
Evaluation over 20 episodes: mean=273.63 +- 66.27

Reference
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}

https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

Potential based rewards (Improve: learning speed)(same optimal solution but avoids local minima)
@inproceedings{10.5555/645528.657613,
author = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart J.},
title = {Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping},
year = {1999},
isbn = {1558606122},
publisher = {Morgan Kaufmann Publishers Inc.},
address = {San Francisco, CA, USA},
booktitle = {Proceedings of the Sixteenth International Conference on Machine Learning},
pages = {278–287},
numpages = {10},
series = {ICML '99}
}