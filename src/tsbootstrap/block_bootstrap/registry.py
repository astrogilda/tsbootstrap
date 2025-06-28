"""
Registry for block bootstrap types.
"""

from typing import Dict, Type

from tsbootstrap.block_bootstrap.base import BlockBootstrap
from tsbootstrap.block_bootstrap.circular import CircularBlockBootstrap
from tsbootstrap.block_bootstrap.moving import MovingBlockBootstrap
from tsbootstrap.block_bootstrap.non_overlapping import (
    NonOverlappingBlockBootstrap,
)
from tsbootstrap.block_bootstrap.stationary import StationaryBlockBootstrap

BLOCK_BOOTSTRAP_TYPES_DICT: Dict[str, Type[BlockBootstrap]] = {
    "nonoverlapping": NonOverlappingBlockBootstrap,
    "moving": MovingBlockBootstrap,
    "stationary": StationaryBlockBootstrap,
    "circular": CircularBlockBootstrap,
}


def get_bootstrap_types_dict() -> Dict[str, Type[BlockBootstrap]]:
    """Get the dictionary of available block bootstrap types."""
    return BLOCK_BOOTSTRAP_TYPES_DICT
