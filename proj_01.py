"""Probabilistyczny wariant Connect Four i prosty eksperyment AI vs AI.

Uruchomienie (wymaga ``uv`` zgodnie z AGENTS.md):
uv run python proj_01/ex_01.py
"""

import random
from typing import Dict, List, Optional, Type

import numpy as np
from easyAI import AI_Player, Negamax, TwoPlayerGame


POS_DIR = np.array(
	[[[i, 0], [0, 1]] for i in range(6)]
	+ [[[0, i], [1, 0]] for i in range(7)]
	+ [[[i, 0], [1, 1]] for i in range(1, 3)]
	+ [[[0, i], [1, 1]] for i in range(4)]
	+ [[[i, 6], [1, -1]] for i in range(1, 3)]
	+ [[[0, i], [1, -1]] for i in range(3, 7)]
)


def find_four(board: np.ndarray, current_player: int) -> bool:
	"""Sprawdza, czy ``current_player`` ma cztery w linii."""

	for pos, direction in POS_DIR:
		streak = 0
		while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
			if board[pos[0], pos[1]] == current_player:
				streak += 1
				if streak == 4:
					return True
			else:
				streak = 0
			pos = pos + direction
	return False


class StandardConnectFour(TwoPlayerGame):
	"""Deterministyczne Connect Four."""

	def __init__(self, players: List[AI_Player], board: Optional[np.ndarray] = None):
		self.players = players
		self.board = board if board is not None else np.zeros((6, 7), dtype=int)
		self.current_player = 1

	def possible_moves(self) -> List[int]:
		return [i for i in range(7) if self.board[:, i].min() == 0]

	def make_move(self, column: int) -> None:
		line = np.argmin(self.board[:, column] != 0)
		self.board[line, column] = self.current_player

	def show(self) -> None:
		print(
			"\n"
			+ "\n".join(
				["0 1 2 3 4 5 6", 13 * "-"]
				+ [
					" ".join([[".", "O", "X"][self.board[5 - j][i]] for i in range(7)])
					for j in range(6)
				]
			)
		)

	def lose(self) -> bool:
		return find_four(self.board, self.opponent_index)

	def is_over(self) -> bool:
		return (self.board.min() > 0) or self.lose()

	def scoring(self) -> int:
		return -100 if self.lose() else 0


class ClumsyConnectFour(StandardConnectFour):
	"""Probabilistyczna wersja: żeton może spaść w kolumnie sąsiedniej."""

	def __init__(
		self,
		players: List[AI_Player],
		board: Optional[np.ndarray] = None,
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
		candidates = {
			intended_column - 1,
			intended_column,
			intended_column + 1,
		}
		candidates = [
			c for c in candidates if 0 <= c <= 6 and self.board[:, c].min() == 0
		]
		if not candidates:
			raise ValueError("No available columns after clumsy drift.")
		return self.rng.choice(candidates)


def _play_single_game(
	game_cls: Type[StandardConnectFour],
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
	game_cls: Type[StandardConnectFour],
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
		res = run_series(StandardConnectFour, depth=depth, games=games_per_setting)
		pretty_print_results("Deterministyczna", depth, res)

	for depth in clumsy_depths:
		res = run_series(ClumsyConnectFour, depth=depth, games=games_per_setting)
		pretty_print_results("Clumsy (prob.)", depth, res)

