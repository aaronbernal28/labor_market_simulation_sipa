import mesa


class SchellingAgent(mesa.Agent):
	"""
	An agent that randomly moves around the transision matrix.
	"""

	def __init__(self, unique_id, model, agent_type):
		super().__init__(unique_id, model)
		self.type = agent_type

	def step(self):
		# 1. Mesa handles the spatial complexity: get all 8 surrounding neighbors
		neighbors = self.model.grid.get_neighbors(
			self.pos, moore=True, include_center=False
		)

		# 2. Calculate local homophily (how many neighbors are like me)
		similar = sum(1 for neighbor in neighbors if neighbor.type == self.type)

		# 3. Decision logic: If I don't have enough similar neighbors, move
		if similar < self.model.homophily:
			self.model.grid.move_to_empty(self)
		else:
			self.model.happy += 1


class LaborMarketModel(mesa.Model):
	"""
	Model class for the
	"""

	def __init__(self, rate_fired=0.1, rate_vancancies=0.1, seed=28):
		super().__init__(seed=seed)

		self.running = True

	def step(self):
		"""Run one step of the model."""
		# TODO
