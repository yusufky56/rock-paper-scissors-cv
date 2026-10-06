# Rock Paper Scissors with Hand Tracking

![CI](https://github.com/yusufky56/rock-paper-scissors-cv/actions/workflows/ci.yml/badge.svg)

Play rock, paper, scissors against the computer using your webcam. **MediaPipe Hands** finds 21 hand landmarks in each frame, the game decides which fingers are extended and turns that into a move.

## How it works

1. **Hand tracking:** MediaPipe returns 21 landmarks per hand (wrist, knuckles, finger joints and tips).
2. **Finger state:** a finger counts as extended when its tip is above its middle joint (PIP).
3. **Gesture:** the number and combination of extended fingers is mapped to a move:

   | Fingers up | Move |
   |---|---|
   | 0–1 | Rock |
   | 2, or index + middle + one more | Scissors |
   | 3 without index + middle, or 4 | Paper |

4. **Smoothing:** the last 5 predictions are kept and the most common one is used, so a single noisy frame doesn't change the move.
5. **Round:** when a hand appears, the computer picks a random move, the winner is shown and the score is updated. Remove your hand to start the next round.

Every second frame is processed and MediaPipe runs with its lightest model, which keeps the game smooth on a laptop CPU.

## Running

```bash
pip install -r requirements.txt
python rps_game.py
```

Show your hand to the camera, press `Esc` to quit.

## Project structure

| File | Content |
|---|---|
| `rps_game.py` | Camera loop, MediaPipe hand tracking, finger detection and the on-screen interface |
| `rps_logic.py` | Gesture rules and winner calculation, no camera needed |
| `tests/test_logic.py` | Unit tests for the rules |

```bash
pip install pytest
pytest
```
