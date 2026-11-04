"""Repository-relative locations used by the experiment scripts."""
import os

EXP = os.path.dirname(os.path.abspath(__file__))   # experiments/ (cache and scratch outputs)
ROOT = os.path.dirname(EXP)                          # repository root (data/, model/, results/, scripts/)
