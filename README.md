# Setup:

1. Create a new python virtual env ``` python .-m venv .venv```.
2. install requirements ``` pip install -r requirements```.

## Lunar Lander Gym

To run the lunar lander env solved with the DQN algorithm: 

- With the already trained model: ``` python dqn_lunar_lander.py```
- With a new model: ``` python dqn_lunar_lander.py new```

![Lunar lander](/gifs/lunar-lander.gif)

## Lunar lander modified

To achieve the opposite action to the original lunar lander, I've placed lander on the heliport coordinates and also modify some values in the exisitng heuristic to prioritize the power on of the main engine.

![Lunar lander modified](/gifs/anti-lunar-lander.gif)

To run the lunar lander version where the lander is taken off run:

- ``` python anti_lunar_lander_run.py ```


