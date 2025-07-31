This is an easily runnable chess engine with GUI in python.



Installation Guide:

1. Clone the repo or download the files.

2. Open the folder in visual studio code or pycharm or your favourite IDE.

3. Download the necessary libraries including pygame, chess. In Pycharm: Right click on pygame or chess and then click to download the necessary libraries, In Visual Studio Code: Use pip install pygame, pip install chess

4. Run the code. It should work :)


How It Works

The Dragon Engine uses a Minimax algorithm with Alpha-Beta Pruning to find optimal moves. It's enhanced with Quiescence Search to stabilize evaluations, a Transposition Table for speed, and Move Ordering to optimize search efficiency. It also incorporates Piece-Square Tables for positional understanding and an Opening Book for strong starts. The game handles pawn promotions and detects draw conditions like three-fold repetition.



For Builders

This project is open source under the MIT License. Feel free to explore, modify, and contribute to enhance the AI, refine the UI, or add new features. Your contributions are welcome!

You can also find a website version here: https://dragon-chess-engine-v7.netlify.app/
