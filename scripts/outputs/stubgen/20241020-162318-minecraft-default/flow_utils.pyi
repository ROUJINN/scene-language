from type_utils import Shape
from typing import Callable

__all__ = ['loop']

def loop(n: int, fn: Callable[[int], Shape]) -> Shape:
    """
    Simple loop executing a function `n` times and concatenating the results.

    Args:
        n (int): Number of iterations.
        fn (Callable[[int], Shape]): Function that takes the current iteration index returns a shape.

    Returns:
        Concatenated shapes from each iteration.
    """
