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


class TestPhiloxStream:
    def test_words_follow_the_counter_layout(self):
        stream = pk.PhiloxStream(0x11111111, 0x22222222, series=(3 << 32) | 9)
        words = [stream.next_word() for _ in range(8)]
        expected = []
        for blk in range(2):
            expected.extend(pk.philox4x32_10((blk, 0, 9, 3), (0x22222222, 0x11111111)))
        assert words == expected

    def test_bounded_rejects_the_biased_low_words(self):
        # For n = 2**32 - 1 the accept threshold is 1: only a word whose product's low half
        # is 0 is rejected, so the output equals the high half of word * n.
        stream = pk.PhiloxStream(1, 2)
        twin = pk.PhiloxStream(1, 2)
        n = 2**32 - 1
        for _ in range(64):
            assert stream.bounded(n) == (twin.next_word() * n) >> 32

    def test_bounded_power_of_two_is_the_top_bits(self):
        stream = pk.PhiloxStream(3, 4)
        twin = pk.PhiloxStream(3, 4)
        for _ in range(64):
            assert stream.bounded(8) == twin.next_word() >> 29

    def test_stream_id(self):
        assert pk.STREAM_ID == "philox4x32-10/v2"
