import pandas as pd

from utils.data_analysis import compute_speedups


class TestDataAnalysis:

    def test_compute_speedups(self):
        data = pd.DataFrame([
            {'id1': 1, 'id2': 1, 'val1': 1, 'val2': 1},
            {'id1': 1, 'id2': 2, 'val1': 2, 'val2': 1},
            {'id1': 2, 'id2': 1, 'val1': 1, 'val2': 2},
            {'id1': 2, 'id2': 2, 'val1': 2, 'val2': 2}
            ]).set_index(['id1', 'id2'])

        speedups = compute_speedups(data, (1), ('id1'))
        assert speedups['val1'].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert speedups['val2'].tolist() == [1.0, 1.0, 0.5, 0.5]
        speedups = compute_speedups(data, (2), ('id2'))
        assert speedups['val1'].tolist() == [2.0, 1.0, 2.0, 1.0]
        assert speedups['val2'].tolist() == [1.0, 1.0, 1.0, 1.0]
