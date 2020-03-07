import gym
import numpy as np

def process_observation(observation):
    '''
    :param observation: numpy array of shape (96, 96, 3) return by the environment
    :return: 96 x 96 grayscale image
    '''
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return observation.dot(rgb_weights)

if __name__=="__main__":
    env_name = 'CarRacing-v0'
    env = gym.make(env_name)

    from PIL import Image
    track = env.reset()
    img = Image.fromarray(track, 'RGB')
    img.show()
    print(track.shape)

    gray_image = process_observation(track)
    print(gray_image.shape)
    #gray_image = np.min(255, gray).astype(np.uint8)
    img = Image.fromarray(gray_image)
    img.show()

    '''
    for i in range(5):
        image = env.render(mode='state_pixels')
        print(image.shape)
        res = env.step(env.action_space.sample())
        state, step_reward, done, _ = res
        print(step_reward)
        print(done)
        track = state
    img = Image.fromarray(track, 'RGB')
    img.show()'''
    env.close()

