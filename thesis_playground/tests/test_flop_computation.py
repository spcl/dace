from flop_computation import FlopCount


class TestFlopCount:

    def test_add(self):
        count1 = FlopCount(adds=1, muls=1, divs=2, minmax=1, powers=1)
        count2 = FlopCount(adds=2, divs=1, abs=1, roots=1)
        count3 = count1 + count2
        assert count3.adds == 3
        assert count3.muls == 1
        assert count3.divs == 3
        assert count3.minmax == 1
        assert count3.abs == 1
        assert count3.powers == 1
        assert count3.roots == 1

    def test_mul(self):
        count1 = FlopCount(adds=1, muls=1, divs=2, minmax=1, abs=2, powers=1, roots=1)
        count2 = count1 * 2
        assert count2.adds == 2
        assert count2.muls == 2
        assert count2.divs == 4
        assert count2.minmax == 2
        assert count2.abs == 4
        assert count2.powers == 2
        assert count2.roots == 2
        count3 = 3 * count1
        assert count3.adds == 3
        assert count3.muls == 3
        assert count3.divs == 6
        assert count3.minmax == 3
        assert count3.abs == 6
        assert count3.powers == 3
        assert count3.roots == 3
