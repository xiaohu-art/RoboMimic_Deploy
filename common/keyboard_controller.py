from pynput import keyboard

class KeyboardController:
    def __init__(self):
        self.keys_pressed = set()
        self.keys_released = set()
        self.previous_keys = set()
        
        self.axis_states = {
            0: 0.0, # LX
            1: 0.0, # LY
            2: 0.0, # RX
            3: 0.0, # RY
        }
        
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
                self.keys_pressed.add(key.char)
            else:
                self.keys_pressed.add(key)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            k = None
            if hasattr(key, 'char') and key.char is not None:
                k = key.char
            else:
                k = key
                
            if k in self.keys_pressed:
                self.keys_pressed.remove(k)
        except AttributeError:
            pass

    def update(self):
        # Calculate just released keys
        self.keys_released = self.previous_keys - self.keys_pressed
        self.previous_keys = self.keys_pressed.copy()
        
        # Axis Logic
        # LX/LY: Arrow Keys
        # RY: Q/E
        
        self.axis_states[1] = 0.0
        self.axis_states[0] = 0.0
        self.axis_states[3] = 0.0
        
        if keyboard.Key.up in self.keys_pressed: self.axis_states[1] -= 1.0
        if keyboard.Key.down in self.keys_pressed: self.axis_states[1] += 1.0
        if keyboard.Key.left in self.keys_pressed: self.axis_states[0] -= 1.0
        if keyboard.Key.right in self.keys_pressed: self.axis_states[0] += 1.0
        if 'q' in self.keys_pressed: self.axis_states[3] -= 1.0
        if 'e' in self.keys_pressed: self.axis_states[3] += 1.0
        
        # Normalize/Clamp if needed, but simple -1/0/1 is usually fine for this

    def is_pressed(self, key):
        return key in self.keys_pressed

    def is_released(self, key):
        return key in self.keys_released
        
    def get_axis_value(self, axis_id):
        return self.axis_states.get(axis_id, 0.0)

    @property
    def Key(self):
        return keyboard.Key
