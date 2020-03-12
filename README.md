# Platform/system and installation versions used to run the code:  

Code was run on macOS Catalina (PyCharm) and Windows 10 Pro 1909 (Anaconda-Sypder) using Python 3.7.

# Packages, dependencies and versions:  
A full list of packages can be found in requirements.txt  
```bash
python==3.7  
tensorflow==2.0.0b1  
tensorboard==2.0.0  
gym==0.17.0  
keras-rl2==1.0.3  
```

### Install order:  
```bash
conda install python=3.7  
pip install gym  
pip install gym[all]  
pip install keras-rl2 # Will install the correct compatible version of TensorFlow (2.0.0b1)  
pip install tensorboard==2.0.0  
```


# The sequence of how the code needs to be executed: 
### To run the code: 
```bash
python src/car_racing.py  
```

### Available arguments:  
```bash
--mode # choices=['train', 'test', 'record'] # Record will make a video of the runs
--window_length # Length of the experience replay window.
--memory_limit  # Limit of how many observations, action, rewards and terminal states to store.
--warmup_steps  # Number of warm-up steps before learning occurs.
--target_model_update  # Controls how often the target network is updated (n'th step). 
--learning_rate  # Set the learning rate.
--train_interval  # How many steps before the model re-fits the neural network
--steps  # Number of total training steps.
--evaluation_episodes  # Total number of test episodes.
--load_weights_from  # Load weights from a previous run.
```

**Training the Model**  
For example, the following arguments will train the model with a limit of 10,000 steps, with 500 warmup steps, and loading weights from a previous run/model. 
```bash
python src/car_racing.py --steps=10000 --warmup_steps=500 --load_weights_from=model_1000 --mode=train
```

**Testing the Model**
```bash
python src/car_racing.py --load_weights_from=model_1000 --evaluation_episodes=10 --mode=test 
```

### TensorBoard:
To run TensorBoard, run the following command on a new python terminal. TensorBoard files can be found in /tensorboard.
```bash
tensorboard --logdir=tensorboard 
```

### Human control:
The following code will allow a human player to play the game:  
```bash
python -m gym.envs.box2d.car_racing  
```
**Controls:**  
Up Arrow - Accelerate  
Down Arrow - Brake   
Left Arrow - Left turn  
Right Arrow - Right turn  
