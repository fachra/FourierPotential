"""Fourier representation of the diffusion MRI signal using layer potentials."""

from .geom import (
    project_stl,
    Ellipse,
    Circle,
    Shape
)

from .core import (
    Solver,
    save,
    load
)

from ._version import __version__

from . import example
