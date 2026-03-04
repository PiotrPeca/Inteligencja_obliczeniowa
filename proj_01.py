import random
from typing import Dict, Optional, Type

from easyAI import AI_Player, Negamax
from easyAI.games.ConnectFour import ConnectFour


class ClumsyConnectFour(ConnectFour):

	def __init__(
		self,
		players,
		board=None,
		rng: Optional[random.Random] = None,
	):
		super().__init__(players, board=board)
		self.rng = rng or random.Random()
		self.last_actual_column: Optional[int] = None

	def make_move(self, column: int) -> None:
		actual_column = self._resolve_column(int(column))
		self.last_actual_column = actual_column
		super().make_move(actual_column)

	def _resolve_column(self, intended_column: int) -> int:
		if self.rng.random() < 0.2:
			neighbors = [intended_column - 1, intended_column + 1]
			self.rng.shuffle(neighbors)
			for col in neighbors:
				if 0 <= col <= 6 and self.board[:, col].min() == 0:
					return col
		return intended_column


def _play_single_game(
	game_cls: Type[ConnectFour],
	depth: int,
	starting_order: str,
	seed: int,
) -> Optional[str]:
	rng = random.Random(seed)
	ai_a = AI_Player(Negamax(depth), name="AI_A")
	ai_b = AI_Player(Negamax(depth), name="AI_B")
	players = [ai_a, ai_b] if starting_order == "A" else [ai_b, ai_a]

	game_kwargs = {"rng": rng} if issubclass(game_cls, ClumsyConnectFour) else {}
	game = game_cls(players, **game_kwargs)
	game.play(nmoves=1000, verbose=False)

	if game.lose():
		winner_player = players[game.opponent_index - 1].name
		return winner_player
	return None  # remis


def run_series(
	game_cls: Type[ConnectFour],
	depth: int,
	games: int,
	base_seed: int = 0,
) -> Dict[str, int]:
	results = {"AI_A": 0, "AI_B": 0, "draw": 0}
	for idx in range(games):
		starting_order = "A" if idx % 2 == 0 else "B"
		seed = base_seed + idx
		winner = _play_single_game(game_cls, depth=depth, starting_order=starting_order, seed=seed)
		if winner is None:
			results["draw"] += 1
		else:
			results[winner] += 1
	return results


def pretty_print_results(header: str, depth: int, results: Dict[str, int]) -> None:
	total = sum(results.values())
	print(f"{header} | depth={depth} | games={total}")
	print(f"  AI_A wins: {results['AI_A']}")
	print(f"  AI_B wins: {results['AI_B']}")
	print(f"  draws   : {results['draw']}\n")


if __name__ == "__main__":
	deterministic_depths = [2, 3]
	clumsy_depths = [2, 3]
	games_per_setting = 20

	for depth in deterministic_depths:
		res = run_series(ConnectFour, depth=depth, games=games_per_setting)
		pretty_print_results("Deterministyczna", depth, res)

	for depth in clumsy_depths:
		res = run_series(ClumsyConnectFour, depth=depth, games=games_per_setting)
		pretty_print_results("Clumsy (prob.)", depth, res)

