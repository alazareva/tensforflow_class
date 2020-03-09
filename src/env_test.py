import gym
import numpy as np

def process_observation(observation):
    '''
    :param observation: numpy array of shape (96, 96, 3) return by the environment
    :return: 96 x 96 grayscale image
    '''
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return observation.dot(rgb_weights)


def compute_steering_speed_gyro_abs(bottom_bar):

    right_steering = bottom_bar[6, 36:46].mean() / 255
    left_steering = bottom_bar[6, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = bottom_bar[6, 46:60].mean() / 255
    right_gyro = bottom_bar[6, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = bottom_bar[:, 0][:-2].mean() / 255
    abs1 = bottom_bar[:, 6][:-2].mean() / 255
    abs2 = bottom_bar[:, 8][:-2].mean() / 255
    abs3 = bottom_bar[:, 10][:-2].mean() / 255
    abs4 = bottom_bar[:, 12][:-2].mean() / 255

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

def grayscale_img(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

def get_bottom_bar(img):
    h, w, _ = img.shape
    s = int(w / 40.0)
    h = h / 40.0
    bottom = grayscale_img(img[84:, :])
    threshold = 20
    black_and_white = (bottom > threshold).astype('uint8') * 255
    x_start = 5 * s
    x_end = (x_start + 1) * s
    speed = black_and_white[:, x_start: x_end].mean()

    x_start = 7 * s
    x_end = (x_start + 1) * s
    wheel_0 = black_and_white[:, x_start: x_end].mean()

    x_start = 8 * s
    x_end = (x_start + 1) * s
    wheel_1 = black_and_white[:, x_start: x_end].mean()

    x_start = 9 * s
    x_end = (x_start + 1) * s
    wheel_2 = black_and_white[:, x_start: x_end].mean()

    x_start = 10 * s
    x_end = (x_start + 1) * s
    wheel_3 = black_and_white[:, x_start: x_end].mean()

    angle_mult = 0.8
    x_start = int((20 - 10 * angle_mult)) * s
    x_end = 20 * s
    angle = black_and_white[:, x_start: x_end].mean()

    velocity_mul = 10 # todo look this up
    x_start = int((20 - 0.8 * velocity_mul)) * s
    x_end = 20 * s
    velocity = black_and_white[:, x_start: x_end].mean()

    return speed, wheel_0, wheel_1, wheel_2, wheel_3, angle, velocity

from gym.envs.box2d.car_racing import CarRacing

if __name__=="__main__":
    env_name = 'CarRacing-v0'
    env = gym.make(env_name)

    from PIL import Image
    track = env.reset()

    for i in range(5):
        print(i)
        track, _, _, _ = env.step(env.action_space.sample())
        bottom = grayscale_img(track)
        threshold = 150
        black_and_white = (bottom > threshold).astype('uint8') * 255
        img = Image.fromarray(bottom)
        img.show()

    #bottom_black_bar = get_bottom_bar(track)
    #print(bottom_black_bar.shape)
    #img = Image.fromarray(bottom_black_bar)
    #img.show()
    #orig = Image.fromarray(track[84:, :], 'RGB')
    #orig.show()
    #print(track.shape)

   # gray_image = process_observation(track)
    #print(gray_image.shape)
    #gray_image = np.min(255, gray).astype(np.uint8)
    #img = Image.fromarray(gray_image)
    #img.show()

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

