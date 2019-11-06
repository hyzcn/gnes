


from .base import BaseService as BS, MessageHandler
from ..proto import gnes_pb2


class PreprocessorService(BS):
    handler = MessageHandler(BS.handler)

    def post_init(self):
        from ..preprocessor.base import BasePreprocessor
        self._model = self.load_model(BasePreprocessor)

    @handler.register(gnes_pb2.Request.TrainRequest)
    def _handler_train(self, msg: 'gnes_pb2.Message'):
        for d in msg.request.train.docs:
            self._apply(d)

    @handler.register(gnes_pb2.Request.IndexRequest)
    def _handler_index(self, msg: 'gnes_pb2.Message'):
        for d in msg.request.index.docs:
            self._apply(d)

    @handler.register(gnes_pb2.Request.QueryRequest)
    def _handler_query(self, msg: 'gnes_pb2.Message'):
        self._apply(msg.request.search.query)

    def _apply(self, d: 'gnes_pb2.Document'):
        self._model.apply(d)
        if not d.chunks:
            self.logger.warning('document (doc_id=%s) contains no chunks!' % d.doc_id)
