


import grpc

from .base import BaseService as BS, MessageHandler
from ..helper import PathImporter
from ..proto import gnes_pb2


class GRPCService(BS):
    handler = MessageHandler(BS.handler)

    def post_init(self):
        self.channel = grpc.insecure_channel(
            '%s:%s' % (self.args.grpc_host, self.args.grpc_port),
            options=[('grpc.max_send_message_length', self.args.max_message_size),
                     ('grpc.max_receive_message_length', self.args.max_message_size)])

        m = PathImporter.add_modules(self.args.pb2_path, self.args.pb2_grpc_path)

        # build stub
        self.stub = getattr(m, self.args.stub_name)(self.channel)

    def close(self):
        self.channel.close()
        super().close()

    @handler.register(NotImplementedError)
    def _handler_default(self, msg: 'gnes_pb2.Message'):
        yield getattr(self.stub, self.args.api_name)(msg)
