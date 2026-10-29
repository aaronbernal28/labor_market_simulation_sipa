import unittest

from src.pipeline.batch_utils import TimeBatch, TimeBatchPartitioner


class TestBatchUtils(unittest.TestCase):
	def test_add_months_same_year(self):
		res = TimeBatchPartitioner._add_months(199601, 5)
		self.assertEqual(res, 199606)

	def test_add_months_year_rollover(self):
		res = TimeBatchPartitioner._add_months(199601, 11)
		self.assertEqual(res, 199612)
		res2 = TimeBatchPartitioner._add_months(199601, 12)
		self.assertEqual(res2, 199701)

	def test_partitioner_annual(self):
		p = TimeBatchPartitioner(start=199601, end=199812, interval=12)
		batch_ids = p.batch_ids()
		self.assertEqual(
			batch_ids,
			["199601_199701", "199701_199801", "199801_199812"],
		)
		self.assertEqual(len(p.batches()), 3)
		self.assertEqual(p.batches()[0], TimeBatch(start=199601, end=199701))

	def test_partitioner_continuous(self):
		p = TimeBatchPartitioner(start=199601, end=202112, interval=12)
		batches = p.batches()
		self.assertGreater(len(batches), 1)
		for k in range(len(batches) - 1):
			self.assertEqual(
				batches[k].end,
				batches[k + 1].start,
				f"Batch {batches[k]} does not connect to {batches[k + 1]}",
			)
		self.assertEqual(batches[0].start, 199601)
		self.assertEqual(batches[-1].end, 202112)

	def test_partitioner_consecutive_pairs(self):
		p = TimeBatchPartitioner(start=199601, end=199812, interval=12)
		pairs = p.consecutive_pairs()
		self.assertEqual(len(pairs), 2)
		self.assertEqual(pairs[0][0].batch_id, "199601_199701")
		self.assertEqual(pairs[0][1].batch_id, "199701_199801")

	def test_partitioner_capped_at_end(self):
		p = TimeBatchPartitioner(start=199601, end=199605, interval=12)
		batch_ids = p.batch_ids()
		self.assertEqual(batch_ids, ["199601_199605"])


if __name__ == "__main__":
	unittest.main()
