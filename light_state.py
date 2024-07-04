_current_light_state = 'green'  # Default state

def get_current_light_state():
    return _current_light_state

def set_current_light_state(state):
    global _current_light_state
    _current_light_state = state