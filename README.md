Open Source Dragon Chess Engine
This is an easily runnable chess engine with GUI in Python. Play against the computer and observe its strategic moves.

Installation Guide
Clone the repo or download the files.

Open the folder in Visual Studio Code, PyCharm, or your favorite IDE.

Download the necessary libraries, including pygame and chess.

In PyCharm: Right-click on pygame or chess and then click to download the necessary libraries.

In Visual Studio Code: Use pip install pygame, pip install chess.

Run the code. It should work! :)

How It Works
The Dragon Engine uses a Minimax algorithm with Alpha-Beta Pruning to find optimal moves. It's enhanced with Quiescence Search to stabilize evaluations, a Transposition Table for speed, and Move Ordering to optimize search efficiency. It also incorporates Piece-Square Tables for positional understanding and an Opening Book for strong starts. The game handles pawn promotions and detects draw conditions like three-fold repetition.

For Builders
This project is open source under the MIT License. Feel free to explore, modify, and contribute to enhance the AI, refine the UI, or add new features. Your contributions are welcome!
