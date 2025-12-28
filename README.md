commands to run:
source .venv/bin/activate

pkill -f tensorboard || true
tensorboard --logdir runs/tb


**Results**
bipedalwalker_baseline_V1:
Evaluation over 10 episodes: mean=202.46 +- 146.54
Evaluation over 20 episodes: mean=284.15 +- 55.43

bipedalwalker_potential_v1:
Evaluation over 10 episodes: mean=296.78 +- 2.69
Evaluation over 20 episodes: mean=273.63 +- 66.27

bipedalwalker_potential_v2:
Evaluation over 20 episodes: mean=173.61 +- 137.29 (first run)
Evaluation over 20 episodes: mean=256.16 +- 83.03

Evaluation of bipedalwalker_potential_v3 over 20 episodes: mean=284.11 +- 34.98
Evaluation of bipedalwalker_potential_v4 over 20 episodes: mean=243.71 +- 102.39
Evaluation of bipedalwalker_potential_v5 over 20 episodes: mean=249.73 +- 100.60
Evaluation of bipedalwalker_potential_v6 over 20 episodes: mean=44.76 +- 136.18
Evaluation of bipedalwalker_potential_v7 over 20 episodes: mean=264.66 +- 91.59
**NOTES:**
- Potential-based reward shaping led to faster early learning and more stable training dynamics, as evidenced by earlier increases in episode reward and length shown by the tensorboard graphs
- The baseline exhibited late-stage performance collapse, whereas the potential model maintained stable episode lengths

**Reference**
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


@misc{andrychowicz2020mattersonpolicyreinforcementlearning,
      title={What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study}, 
      author={Marcin Andrychowicz and Anton Raichuk and Piotr Stańczyk and Manu Orsini and Sertan Girgin and Raphael Marinier and Léonard Hussenot and Matthieu Geist and Olivier Pietquin and Marcin Michalski and Sylvain Gelly and Olivier Bachem},
      year={2020},
      eprint={2006.05990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.05990}, 
}