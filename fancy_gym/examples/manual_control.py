import keyboard
import numpy as np

class ManualControl:
    def __init__(self, action_shape):
        self.action = np.zeros(action_shape)
        self.setup_key_listeners()

    def setup_key_listeners(self):
        keyboard.on_press_key("left", lambda _: self.update_action(0, -1))
        keyboard.on_press_key("right", lambda _: self.update_action(0, 1))
        keyboard.on_press_key("up", lambda _: self.update_action(1, 1))
        keyboard.on_press_key("down", lambda _: self.update_action(1, -1))

        keyboard.on_release_key("left", lambda _: self.reset_action(0))
        keyboard.on_release_key("right", lambda _: self.reset_action(0))
        keyboard.on_release_key("up", lambda _: self.reset_action(1))
        keyboard.on_release_key("down", lambda _: self.reset_action(1))

    def update_action(self, index, value):
        self.action[index] = value

    def reset_action(self, index):
        self.action[index] = 0
