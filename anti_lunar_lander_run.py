import gym
import numpy as np


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
        a: action to execute (nothing, up, left, right).
    """

    # I changed the heuristic to prioritze the power on in the main engine
    angle_targ = s[0]*0.5 + s[3]*1.0
    if angle_targ > 0:
        angle_targ = -1
    if angle_targ <= 0:
        angle_targ = 1
    hover_targ = np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
    hover_todo = (hover_targ + s[1])*0.5 + (s[3])*0.5

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < 0:
        a = 3
    elif angle_todo > 0:
        a = 1
    return a


def main(env, render=True):
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    
    return total_reward


if __name__ == '__main__':
    env = gym.make("AntiLunarLander-v0")
    main(env)
