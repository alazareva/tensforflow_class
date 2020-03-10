# Platform/system and installation versions used to run the code:  

Code was run on macOS Catalina (PyCharm) and Windows 10 Pro 1909 (Anaconda-Sypder) using Python 3.7.

# Packages, dependencies and versions:  
A full list of packages can be found in requirements.txt  
python==3.7  
tensorflow==2.0.0b1  
tensorboard==2.0.0  
gym==0.17.0  
keras-rl2==1.0.3  


### Install order:  
conda install python=3.7  
pip install gym  
pip install gym[all]  
pip install keras-rl2 #Will install the correct compatible version of TensorFlow (2.0.0b1)  
pip install tensorboard==2.0.0  



# The sequence of how the code needs to be executed: 
### To run the code: 

python src/car_racing.py  


### Available arguments:  

--mode # choices=['train', 'test']
--window_length # The length of the filter window. Must be a positive integer
--memory_limit  # In bytes (?????)
--warmup_steps  # Lower learning rate during the warmup steps.
--target_model_update  # Controls how often the target network is updated (n'th step). 
--learning_rate  # Set the learning rate.
--train_interval  # How many steps before the model re-fits the neural network
--steps  # Number of total training steps
--evaluation_episodes  # ?????
--load_weights_from  # Load weights from a previous run.


For example, the following arguments will run the code with a limit of 10,000 steps, with 500 warmup steps during which the learning rate is lowered, and loading weights from a previous run/model. 

python src/car_racing.py  --steps=10000 --warmup_steps=500 --load_weights_from=pretrained_model_v1

# Any other details that you think might be useful
Do we need to cite pieces of work we used/incorporated/found helpful?