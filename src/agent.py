import mesa


class SchellingAgent(mesa.Agent):
	"""
	An agent that randomly moves around the transision matrix.
	"""

	def __init__(
		self, unique_id: int, model: mesa.Model, agent_type: int | str
	) -> None:
		super().__init__(unique_id, model)
		self.type: int | str = agent_type

	def step(self) -> None:
		# 1. Mesa handles the spatial complexity: get all 8 surrounding neighbors
		neighbors: list = self.model.grid.get_neighbors(
			self.pos, moore=True, include_center=False
		)

		# 2. Calculate local homophily (how many neighbors are like me)
		similar: int = sum(1 for neighbor in neighbors if neighbor.type == self.type)

		# 3. Decision logic: If I don't have enough similar neighbors, move
		if similar < self.model.homophily:
			self.model.grid.move_to_empty(self)
		else:
			self.model.happy += 1


class LaborMarketModel(mesa.Model):
	"""
	Model class for the
	"""

	def __init__(
		self,
		rate_fired: float = 0.1,
		rate_vancancies: float = 0.1,
		seed: int = 28,
	) -> None:
		super().__init__(seed=seed)

		self.running: bool = True

	def step(self) -> None:
		"""Run one step of the model."""
		# TODO
