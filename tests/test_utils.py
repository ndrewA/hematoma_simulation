"""Tests for preprocessing/utils.py — grid construction, resampling, structuring elements."""

import argparse

import numpy as np
import pytest

from preprocessing.utils import (
    PROFILES, add_grid_args, build_grid_affine, resample_to_grid,
    resolve_grid_args, build_ball,
    raw_dir, processed_dir, validation_dir,
)


# ---------------------------------------------------------------------------
# build_grid_affine
# ---------------------------------------------------------------------------
class TestBuildGridAffine:
    @pytest.mark.parametrize("N,dx", [p for p in PROFILES.values()])
    def test_center_maps_to_origin(self, N, dx):
        A = build_grid_affine(N, dx)
        center = np.array([N / 2, N / 2, N / 2, 1.0])
        phys = A @ center
        np.testing.assert_allclose(phys[:3], 0.0, atol=1e-12)

    @pytest.mark.parametrize("N,dx", [p for p in PROFILES.values()])
    def test_diagonal_positive_ras(self, N, dx):
        A = build_grid_affine(N, dx)
        assert A[0, 0] > 0 and A[1, 1] > 0 and A[2, 2] > 0

    def test_voxel_spacing(self):
        A = build_grid_affine(256, 1.0)
        assert A[0, 0] == 1.0
        assert A[1, 1] == 1.0
        assert A[2, 2] == 1.0

    def test_physical_extent(self):
        N, dx = 256, 1.0
        A = build_grid_affine(N, dx)
        # Voxel 0 should map to -128mm, voxel 255 to +127mm
        origin_phys = A @ np.array([0, 0, 0, 1.0])
        np.testing.assert_allclose(origin_phys[:3], -128.0)
        far_phys = A @ np.array([255, 255, 255, 1.0])
        np.testing.assert_allclose(far_phys[:3], 127.0)


# ---------------------------------------------------------------------------
# build_ball
# ---------------------------------------------------------------------------
class TestBuildBall:
    def test_shape(self):
        ball = build_ball(3)
        assert ball.shape == (7, 7, 7)

    def test_center_true(self):
        ball = build_ball(3)
        assert ball[3, 3, 3]

    def test_corners_false(self):
        ball = build_ball(3)
        assert not ball[0, 0, 0]
        assert not ball[6, 6, 6]

    def test_symmetric(self):
        ball = build_ball(4)
        # Should be symmetric under all axis flips
        assert np.array_equal(ball, ball[::-1, :, :])
        assert np.array_equal(ball, ball[:, ::-1, :])
        assert np.array_equal(ball, ball[:, :, ::-1])

    def test_radius_1(self):
        ball = build_ball(1)
        assert ball.shape == (3, 3, 3)
        assert ball[1, 1, 1]
        # Face neighbors should be True (distance = 1)
        assert ball[0, 1, 1]
        # Edge neighbors should NOT be True (distance = sqrt(2) > 1)
        assert not ball[0, 0, 1]


# ---------------------------------------------------------------------------
# resample_to_grid
# ---------------------------------------------------------------------------
class TestResampleToGrid:
    def test_identity(self):
        src = np.arange(8**3, dtype=np.float64).reshape(8, 8, 8)
        result = resample_to_grid((src, np.eye(4)), np.eye(4), (8, 8, 8),
                                  order=1)
        np.testing.assert_allclose(result, src)

    def test_identity_integer(self):
        src = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int16)
        result = resample_to_grid((src, np.eye(4)), np.eye(4), (2, 2, 2),
                                  order=0, dtype=np.int16)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, src)

    def test_shifted(self):
        src = np.arange(8**3, dtype=np.float64).reshape(8, 8, 8)
        shifted = np.eye(4)
        shifted[:3, 3] = 1.0  # grid origin shifted +1mm
        result = resample_to_grid((src, np.eye(4)), shifted, (8, 8, 8),
                                  order=0, cval=-1.0, dtype=np.float64)
        # Grid voxel (0,0,0) -> physical (1,1,1) -> source voxel (1,1,1)
        assert result[0, 0, 0] == src[1, 1, 1]
        # Grid voxel (7,7,7) -> OOB -> cval
        assert result[7, 7, 7] == -1.0

    def test_oob_constant_mode(self):
        src = np.zeros((4, 4, 4), dtype=np.float64)
        src[0, 0, 0] = 99.0
        big_shift = np.eye(4)
        big_shift[:3, 3] = 10.0
        result = resample_to_grid((src, np.eye(4)), big_shift, (4, 4, 4),
                                  order=0, cval=0.0, dtype=np.float64)
        np.testing.assert_array_equal(result, 0.0)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
class TestPathHelpers:
    def test_raw_dir_absolute(self):
        assert raw_dir("157336").is_absolute()

    def test_raw_dir_structure(self):
        p = raw_dir("157336")
        assert p.parts[-3:] == ("raw", "157336", "T1w")

    def test_processed_dir_structure(self):
        p = processed_dir("157336", "dev")
        assert p.parts[-3:] == ("processed", "157336", "dev")

    def test_validation_dir_structure(self):
        p = validation_dir("157336")
        assert p.parts[-2:] == ("validation", "157336")


# ---------------------------------------------------------------------------
# add_grid_args / resolve_grid_args
# ---------------------------------------------------------------------------
def _build_parser():
    """Build a fresh argparse parser with grid args."""
    parser = argparse.ArgumentParser()
    add_grid_args(parser)
    return parser


class TestAddGridArgs:
    def test_subject_required(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_profile_choices(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234", "--profile", "debug"])
        assert args.profile == "debug"

    def test_profile_invalid_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--subject", "1234", "--profile", "invalid"])

    def test_dx_accepted(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234", "--dx", "0.5",
                                  "--grid-size", "512"])
        assert args.dx == 0.5
        assert args.grid_size == 512

    def test_profile_dx_mutually_exclusive(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--subject", "1234", "--profile", "debug",
                               "--dx", "0.5"])


class TestResolveGridArgs:
    def test_default_is_debug(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234"])
        resolve_grid_args(args, parser)
        assert args.profile == "debug"
        assert args.N == 128
        assert args.dx == 2.0

    def test_profile_prod(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234", "--profile", "prod"])
        resolve_grid_args(args, parser)
        assert args.N == 512
        assert args.dx == 0.5

    def test_custom_dx_grid_size(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234", "--dx", "0.75",
                                  "--grid-size", "300"])
        resolve_grid_args(args, parser)
        assert args.N == 300
        assert args.dx == 0.75
        assert args.profile == "custom_300_0.75"

    def test_dx_without_grid_size_errors(self):
        parser = _build_parser()
        args = parser.parse_args(["--subject", "1234", "--dx", "0.5"])
        with pytest.raises(SystemExit):
            resolve_grid_args(args, parser)
