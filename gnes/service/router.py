


from collections import defaultdict
from typing import Dict, List

from .base import BaseService as BS, MessageHandler, BlockMessage
from ..proto import gnes_pb2
from ..router.base import BaseReduceRouter


class RouterService(BS):
    handler = MessageHandler(BS.handler)

    def post_init(self):
        from ..router.base import BaseRouter
        self._model = self.load_model(BaseRouter)
        self._pending = defaultdict(list)  # type: Dict[str, List]

    def _is_msg_complete(self, msg: 'gnes_pb2.Message', num_req: int) -> bool:
        return (self.args.num_part is None and num_req == msg.envelope.num_part[-1]) or \
               (num_req == self.args.num_part)

    @handler.register(NotImplementedError)
    def _handler_default(self, msg: 'gnes_pb2.Message'):
        if isinstance(self._model, BaseReduceRouter):
            req_id = msg.envelope.request_id
            self._pending[req_id].append(msg)
            num_req = len(self._pending[req_id])

            if self._is_msg_complete(msg, num_req):
                prev_msgs = self._pending.pop(req_id)
                return self._model.apply(msg, prev_msgs)
            else:
                raise BlockMessage
        else:
            return self._model.apply(msg)
