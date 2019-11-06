

from .base import get_unary_score, CombinedScoreFn
import json


class WeightedDocScoreFn(CombinedScoreFn):
    def __call__(self, last_score: 'gnes_pb2.Response.QueryResponse.ScoredResult.Score',
                 doc: 'gnes_pb2.Document', *args, **kwargs):
        d_weight = get_unary_score(value=doc.weight,
                                   name='doc weight',
                                   doc_id=doc.doc_id)
        return super().__call__(last_score, d_weight)


class CoordDocScoreFn(CombinedScoreFn):
    """
    score = score * query_coordination
    query_coordination: #chunks recalled / #chunks in this doc
    """

    def __call__(self, last_score: 'gnes_pb2.Response.QueryResponse.ScoredResult.Score',
                 doc: 'gnes_pb2.Document',
                 *args, **kwargs):
        total_chunks = len(doc.chunks)
        recall_chunks = len(json.loads(last_score.explained)['operands'])
        query_coord = 1 if total_chunks == 0 else recall_chunks / total_chunks
        d_weight = get_unary_score(value=query_coord,
                                   name='query coordination')
        return super().__call__(last_score, d_weight)

