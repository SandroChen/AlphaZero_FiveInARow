# Info
This is a project of implementing general AlphaZero which develops from the micro-project about FiveInARow in HKUST Master Course(MSDM5002 Final Collaboration Project).

# Update
2025.1.6
1. Update MCTS algorithm and add a new file myts.py. Now the MCTS part is more like the original MCTS in alphago zero paper, which is still different from the one in paper:
   1) we temporarily use uct instead of pust, because the mix usage of mcts and network has not been implemented.
   2) In PLAY section, the method in paper chooses move using visit count, our implementation uses mean action value instead.
2. Add Neural Network in models.py, which has the same architecture with the network used in alphago zero.

# Install
python >= 3.8
```
pip install -r requirement.txt
```


# Play
play game in 3x3 board using pure mcts
```
python main.py
```
play game in 3x3 board using pure network
```
python main.py --method 1 --ckpt ckpts/33_iter100/MovePredictor_4999.pth
```


# Done: 
1. Front-end interface on which human can play against AI based on pure MCTS or pure neural network.
2. Pretrained neural network on 3x3 board(n_in_a_row=3).

# Todo: 
1. Pretrained neural network on 6x6 board(n_in_a_row=4).
2. Pretrained neural network on 8x8 board(n_in_a_row=5).

# Reference
1. D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017). https://doi.org/10.1038/nature24270
2. Third-party open-source implementation：[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
