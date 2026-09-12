"""Transparent synthetic proof-of-concept action augmentation (not clinical truth)."""
from .patient_traits import generate_traits
from .episode_generator import generate_hybrid_episodes
__all__=["generate_traits","generate_hybrid_episodes"]
