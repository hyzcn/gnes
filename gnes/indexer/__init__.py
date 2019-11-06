


# A key-value map for Class to the (module)file it located in
from ..base import register_all_class

_cls2file_map = {
    'FaissIndexer': 'chunk.faiss',
    'LVDBIndexer': 'doc.leveldb',
    'RocksDBIndexer': 'doc.rocksdb',
    'AsyncLVDBIndexer': 'doc.leveldb',
    'NumpyIndexer': 'chunk.numpy',
    'BIndexer': 'chunk.bindexer',
    'HBIndexer': 'chunk.hbindexer',
    'JointIndexer': 'base',
    'BaseIndexer': 'base',
    'BaseDocIndexer': 'base',
    'AnnoyIndexer': 'chunk.annoy',
    'DirectoryIndexer': 'doc.filesys',
    'DictIndexer': 'doc.dict',
    'DictKeyIndexer': 'chunk.helper',
    'ListKeyIndexer': 'chunk.helper',
    'ListNumpyKeyIndexer': 'chunk.helper',
    'NumpyKeyIndexer': 'chunk.helper',
}

register_all_class(_cls2file_map, 'indexer')
