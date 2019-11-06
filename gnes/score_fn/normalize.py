

from .base import ModifierScoreFn, ScoreOps as so


class Normalizer1(ModifierScoreFn):
    """Do normalizing: score = 1 / (1 + sqrt(score))"""

    def __init__(self):
        super().__init__(modifier='reciprocal1p')

    def __call__(self, last_score, *args, **kwargs):
        return super().__call__(so.sqrt(last_score))


class Normalizer2(ModifierScoreFn):
    """Do normalizing: score = 1 / (1 + score / num_dim)"""

    def __init__(self, num_dim: int):
        super().__init__(modifier='reciprocal1p', factor=1.0 / num_dim, factor_name='1/num_dim')


class Normalizer3(Normalizer2):
    """Do normalizing: score = 1 / (1 + sqrt(score) / num_dim)"""

    def __call__(self, last_score, *args, **kwargs):
        return super().__call__(so.sqrt(last_score))


class Normalizer4(ModifierScoreFn):
    """Do normalizing: score = 1 - score / num_bytes """

    def __init__(self, num_bytes: int):
        super().__init__(modifier='invert1p', factor=1.0 / num_bytes, factor_name='1/num_bytes')


class Normalizer5(ModifierScoreFn):
    """Do normalizing: score = 1 / (1 + sqrt(abs(score)))"""

    def __init__(self):
        super().__init__(modifier='reciprocal1p')

    def __call__(self, last_score, *args, **kwargs):
        return super().__call__(so.sqrt(so.abs(last_score)))
