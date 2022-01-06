# udacity_rl_p1

##Project Details

###State space


```

Number of agents: 1
Number of actions: 4
States look like: [1.         0.         0.         0.         0.84408134 0.
0.         1.         0.         0.0748472  0.         1.
0.         0.         0.25755    1.         0.         0.
0.         0.74177343 0.         1.         0.         0.
0.25854847 0.         0.         1.         0.         0.09355672
0.         1.         0.         0.         0.31969345 0.
0.        ]
States have length: 37
```
The environment is considered to be solved if agent gets 13 points in 30 second episode and
the average score is calculated over 100 episodes. 

The solution is described in ```report.md``` file

## Getting Started

Install Python 3.6 and then
```
cd python && pip install -e . 
unzip Banana_Linux_NoVis.zip or unzip Banana_Linux_Vis.zip 
python -m ipykernel install --user --name drlnd --display-name "drlnd"
jupyter notebook
```

## Project sructure

1. ```main.py``` contains the learning algorithm procedure.
2. ```model.py``` requires the network
3. ```dqn_agent``` contains the DQN algorithm
4. ```checkpoints``` directory with saved checkpoints.
5. ```images``` contains graphics.
6. ```p1_navigation_original``` contains original task description
7. ```python``` directory, has required packages
