import random
from time import perf_counter
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


def _negamax_no_ab(game, depth: int, orig_depth: int, scoring) -> float:
	if (depth == 0) or game.is_over():
		return scoring(game) * (1 + 0.001 * depth)

	possible_moves = game.possible_moves()
	state = game
	best_move = possible_moves[0]
	if depth == orig_depth:
		state.ai_move = best_move

	best_value = -float("inf")
	unmake_move = hasattr(state, "unmake_move")

	for move in possible_moves:
		if not unmake_move:
			game = state.copy()

		game.make_move(move)
		game.switch_player()
		move_value = -_negamax_no_ab(game, depth - 1, orig_depth, scoring)

		if unmake_move:
			game.switch_player()
			game.unmake_move(move)

		if best_value < move_value:
			best_value = move_value
			best_move = move
			if depth == orig_depth:
				state.ai_move = move

	return best_value


class NegamaxNoAB:
	"""Negamax bez odcięcia alpha-beta."""

	def __init__(self, depth, scoring=None):
		self.scoring = scoring
		self.depth = depth

	def __call__(self, game):
		scoring = self.scoring if self.scoring else (lambda g: g.scoring())
		self.value = _negamax_no_ab(game, self.depth, self.depth, scoring)
		return game.ai_move


class TimedAIPlayer(AI_Player):
	def __init__(self, AI_algo, name: str, timing_key: str, timing_stats: Dict[str, Dict[str, float]]):
		super().__init__(AI_algo, name=name)
		self.timing_key = timing_key
		self.timing_stats = timing_stats

	def ask_move(self, game):
		start = perf_counter()
		move = super().ask_move(game)
		elapsed = perf_counter() - start
		self.timing_stats[self.timing_key]["total_time_s"] += elapsed
		self.timing_stats[self.timing_key]["moves"] += 1
		return move


def _play_single_game(
	game_cls: Type[ConnectFour],
	depth: int,
	starting_order: str,
	seed: int,
	ai_algo_cls=Negamax,
	timing_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[str]:
	rng = random.Random(seed)
	if timing_stats is None:
		timing_stats = {
			"AI_A": {"total_time_s": 0.0, "moves": 0},
			"AI_B": {"total_time_s": 0.0, "moves": 0},
		}

	ai_a = TimedAIPlayer(ai_algo_cls(depth), name="AI_A", timing_key="AI_A", timing_stats=timing_stats)
	ai_b = TimedAIPlayer(ai_algo_cls(depth), name="AI_B", timing_key="AI_B", timing_stats=timing_stats)
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
	ai_algo_cls=Negamax,
) -> Dict[str, float]:
	results = {"AI_A": 0, "AI_B": 0, "draw": 0}
	timing_stats = {
		"AI_A": {"total_time_s": 0.0, "moves": 0},
		"AI_B": {"total_time_s": 0.0, "moves": 0},
	}
	for idx in range(games):
		starting_order = "A" if idx % 2 == 0 else "B"
		seed = base_seed + idx
		winner = _play_single_game(
			game_cls,
			depth=depth,
			starting_order=starting_order,
			seed=seed,
			ai_algo_cls=ai_algo_cls,
			timing_stats=timing_stats,
		)
		if winner is None:
			results["draw"] += 1
		else:
			results[winner] += 1

	for key in ("AI_A", "AI_B"):
		moves = timing_stats[key]["moves"]
		total = timing_stats[key]["total_time_s"]
		results[f"avg_time_{key}_ms"] = (1000.0 * total / moves) if moves else 0.0

	return results


def pretty_print_results(header: str, depth: int, results: Dict[str, int]) -> None:
	total = sum(results.values())
	print(f"{header} | depth={depth} | games={total}")
	print(f"  AI_A wins: {results['AI_A']}")
	print(f"  AI_B wins: {results['AI_B']}")
	print(f"  draws   : {results['draw']}")
	print(f"  avg move time AI_A: {results['avg_time_AI_A_ms']:.3f} ms")
	print(f"  avg move time AI_B: {results['avg_time_AI_B_ms']:.3f} ms\n")


if __name__ == "__main__":
	depths = [2, 3]
	games_per_setting = 20
	ai_variants = [
		("Negamax (alpha-beta)", Negamax),
		("Negamax (bez alpha-beta)", NegamaxNoAB),
	]
	game_variants = [
		("Deterministyczna", ConnectFour),
		("Clumsy (prob.)", ClumsyConnectFour),
	]

	for game_label, game_cls in game_variants:
		for ai_label, ai_algo_cls in ai_variants:
			for depth in depths:
				res = run_series(game_cls, depth=depth, games=games_per_setting, ai_algo_cls=ai_algo_cls)
				pretty_print_results(f"{game_label} | {ai_label}", depth, res)

