"""Game rules and gesture classification, independent of the camera and MediaPipe."""

MOVES = ('rock', 'paper', 'scissors')

# What each move beats
BEATS = {
    'rock': 'scissors',
    'scissors': 'paper',
    'paper': 'rock',
}


def classify_gesture(finger_states):
    """Map the extended state of [index, middle, ring, pinky] to a move.

    - 0-1 fingers up: rock (a single raised finger is usually a half-closed fist)
    - 2 fingers up: scissors
    - 3 fingers up: scissors if index and middle are up, otherwise paper
    - 4 fingers up: paper
    """
    index_up, middle_up = finger_states[0], finger_states[1]
    extended = sum(bool(s) for s in finger_states)

    if extended <= 1:
        return 'rock'
    if extended == 2:
        return 'scissors'
    if extended == 3:
        return 'scissors' if index_up and middle_up else 'paper'
    return 'paper'


def round_winner(player_move, computer_move):
    """Return 'draw', 'player_victory' or 'computer_victory'."""
    if player_move == computer_move:
        return 'draw'
    if BEATS.get(player_move) == computer_move:
        return 'player_victory'
    return 'computer_victory'
