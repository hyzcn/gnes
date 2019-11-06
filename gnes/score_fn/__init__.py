

# A key-value map for Class to the (module)file it located in
from ..base import register_all_class

_cls2file_map = {
    'BaseScoreFn': 'base',
    'CombinedScoreFn': 'base',
    'ModifierScoreFn': 'base',
    'WeightedChunkScoreFn': 'chunk',
    'WeightedDocScoreFn': 'doc',
    'Normalizer1': 'normalize',
    'Normalizer2': 'normalize',
    'Normalizer3': 'normalize',
    'Normalizer4': 'normalize',
    'Normalizer5': 'normalize',
}

register_all_class(_cls2file_map, 'score_fn')
