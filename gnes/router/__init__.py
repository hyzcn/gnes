


# A key-value map for Class to the (module)file it located in
from ..base import register_all_class

_cls2file_map = {
    'BaseRouter': 'base',
    'BaseMapRouter': 'base',
    'BaseReduceRouter': 'base',
    'BaseTopkReduceRouter': 'base',
    'BaseEmbedReduceRouter': 'base',
    'DocTopkReducer': 'reduce',
    'ChunkTopkReducer': 'reduce',
    'DocFillReducer': 'reduce',
    'PublishRouter': 'map',
    'DocBatchRouter': 'map',
    'ConcatEmbedRouter': 'reduce',
    'AvgEmbedRouter': 'reduce'
}

register_all_class(_cls2file_map, 'router')
