from pycarcassone import Game, RandomPlayer

player1 = RandomPlayer(7153)
player2 = RandomPlayer(3651)
player3 = RandomPlayer(1351)
players = [player1, player2, player3]
game = Game(players, seed=67, enable_render=False)
game.reset()
game.mainloop()

game.render()
game.close()
