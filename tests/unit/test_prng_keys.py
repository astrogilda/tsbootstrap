"""Value pins for the numba-free PRNG key reference (tsbootstrap.prng_keys).

These pin the reference oracle at the unit level: the compiled kernels recompute the same
arithmetic in ``np.uint64`` and the same-stream tests hold them equal to this reference, so
a mutation to any reference function that changed a constant, a shift, or a mix breaks a
literal here. The philox pins are the published Random123 known-answer vectors.
"""

from __future__ import annotations

from tsbootstrap import prng_keys as pk


class TestReplicateKey:
    def test_known_values(self):
        assert pk.replicate_key(0, 0, 0) == (1318458718, 1594498111)
        assert pk.replicate_key(12345, 67890, 3) == (847031815, 294882464)

    def test_injective_in_b(self):
        root_a, root_b = 0xABCDEF, 0x123456
        keys = [pk.replicate_key(root_a, root_b, b) for b in range(64)]
        assert len(set(keys)) == 64

    def test_both_root_halves_matter(self):
        assert pk.replicate_key(5, 0, 1) != pk.replicate_key(6, 0, 1)  # root_a
        assert pk.replicate_key(5, 0, 1) != pk.replicate_key(5, 1, 1)  # root_b (mid-round fold)


class TestSeriesFold:
    def test_hash_slot0_is_identity(self):
        assert pk.hash_series_words(0) == (0, 0)

    def test_hash_known_values(self):
        assert pk.hash_series_words(1) == (2354591315, 3668224091)
        assert pk.hash_series_words(5) == (2984982325, 258174789)

    def test_hash_injective(self):
        pairs = [pk.hash_series_words(s) for s in range(64)]
        assert len(set(pairs)) == 64

    def test_fold_slot0_leaves_key_unchanged(self):
        rk = pk.replicate_key(7, 7, 1)
        assert pk.fold_in_key(*rk, 0) == rk

    def test_fold_known_values(self):
        assert pk.fold_in_key(0, 0, 3) == (447418513, 156322883)
        assert pk.fold_in_key(*pk.replicate_key(7, 7, 1), 2) == (3510559759, 764993004)


class TestPhilox4x32:
    def test_kat_zero(self):
        assert pk.philox4x32_10((0, 0, 0, 0), (0, 0)) == (
            0x6627E8D5,
            0xE169C58D,
            0xBC57AC4C,
            0x9B00DBD8,
        )

    def test_kat_ones(self):
        assert pk.philox4x32_10((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF)) == (
            0x408F276D,
            0x41C83B0E,
            0xA20BC7C6,
            0x6D5451FD,
        )

    def test_kat_pi_digits(self):
        assert pk.philox4x32_10(
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)
        ) == (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)


class TestU01:
    def test_endpoints(self):
        assert pk.u01_from_word(0) == 0.0
        assert pk.u01_from_word(2**31) == 0.5

    def test_in_unit_interval(self):
        assert 0.0 <= pk.u01_from_word(0xFFFFFFFF) < 1.0
