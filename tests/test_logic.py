import itertools

import pytest

from rps_logic import MOVES, classify_gesture, round_winner


@pytest.mark.parametrize("states, expected", [
    ([False, False, False, False], 'rock'),
    ([True, False, False, False], 'rock'),
    ([True, True, False, False], 'scissors'),
    ([False, True, True, False], 'scissors'),
    ([True, True, True, False], 'scissors'),
    ([False, True, True, True], 'paper'),
    ([True, True, True, True], 'paper'),
])
def test_classify_gesture(states, expected):
    assert classify_gesture(states) == expected


def test_every_combination_maps_to_a_move():
    for states in itertools.product([False, True], repeat=4):
        assert classify_gesture(list(states)) in MOVES


@pytest.mark.parametrize("player, computer, result", [
    ('rock', 'scissors', 'player_victory'),
    ('scissors', 'paper', 'player_victory'),
    ('paper', 'rock', 'player_victory'),
    ('scissors', 'rock', 'computer_victory'),
    ('paper', 'scissors', 'computer_victory'),
    ('rock', 'paper', 'computer_victory'),
    ('rock', 'rock', 'draw'),
])
def test_round_winner(player, computer, result):
    assert round_winner(player, computer) == result
