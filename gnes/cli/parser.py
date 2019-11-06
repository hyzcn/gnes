


import argparse


class ActionNoYes(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):

        if default is None:
            raise ValueError('you must provide a default with yes/no action')
        if len(option_strings) != 1:
            raise ValueError('only single argument is allowed with yes/no action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('yes/no arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt, '--no_' + opt]
        super(ActionNoYes, self).__init__(opts, dest, nargs=0, const=None,
                                          default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-') or option_strings.startswith('--no_'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


def resolve_py_path(path):
    import os
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('%s is not a valid file path' % path)
    return path


def random_port(port):
    if not port or int(port) <= 0:
        import random
        min_port, max_port = 49152, 65536
        return random.randrange(min_port, max_port)
    else:
        return int(port)


def resolve_yaml_path(path, to_stream=False):
    # priority, filepath > classname > default
    import os
    import io
    if hasattr(path, 'read'):
        # already a readable stream
        return path
    elif os.path.exists(path):
        if to_stream:
            return open(path, encoding='utf8')
        else:
            return path
    elif path.isidentifier():
        # possible class name
        return io.StringIO('!%s {}' % path)
    elif path.startswith('!'):
        # possible YAML content
        return io.StringIO(path)
    else:
        raise argparse.ArgumentTypeError('%s can not be resolved, it should be a readable stream,'
                                         ' or a valid file path, or a supported class name.' % path)


def set_base_parser():
    from .. import __version__, __proto_version__
    from termcolor import colored
    import os
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description='%s, a cloud-native semantic search system '
                    'based on deep neural network. '
                    'It enables large-scale index and semantic search for text-to-text, image-to-image, '
                    'video-to-video and any content form. Visit %s for tutorials and documentations.' % (
                        colored('GNES v%s: Generic Neural Elastic Search' % __version__, 'green'),
                        colored('https://gnes.ai', 'cyan', attrs=['underline'])),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s' + ': %s\nprotobuf: %s\nvcs_version: %s' %
                                (__version__, __proto_version__, os.environ.get('GNES_VCS_VERSION', 'unknown')))
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on detailed logging for debug')
    return parser


def set_composer_parser(parser=None):
    from pkg_resources import resource_stream

    if not parser:
        parser = set_base_parser()
    parser.add_argument('--port',
                        type=int,
                        default=8800,
                        help='host port of the grpc service')
    parser.add_argument('--name',
                        type=str,
                        default='GNES app',
                        help='name of the instance')
    parser.add_argument('--yaml_path', type=lambda x: resolve_yaml_path(x, True),
                        default=resource_stream(
                            'gnes', '/'.join(('resources', 'compose', 'gnes-example.yml'))),
                        help='yaml config of the service')
    parser.add_argument('--html_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the HTML file, will contain all possible generations')
    parser.add_argument('--shell_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the shell-based starting script')
    parser.add_argument('--swarm_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the docker-compose file for Docker Swarm')
    parser.add_argument('--k8s_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the docker-compose file for Docker Swarm')
    parser.add_argument('--graph_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the mermaid graph file')
    parser.add_argument('--shell_log_redirect', type=str,
                        help='the file path for redirecting shell output. '
                             'when not given, the output will be flushed to stdout')
    parser.add_argument('--mermaid_leftright', action='store_true', default=False,
                        help='showing the flow in left-to-right manner rather than top down')
    parser.add_argument('--docker_img', type=str,
                        default='gnes/gnes:latest-alpine',
                        help='the docker image used in Docker Swarm & Kubernetes')
    return parser


def set_composer_flask_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    set_composer_parser(parser)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--flask', action='store_true', default=False,
                       help='start a Flask server and serve the composer in interactive mode, aka GNES board')
    group.add_argument('--serve', action='store_true', default=False,
                       help='start a basic HTTP server and serve the composer in interactive mode, aka GNES board')
    parser.add_argument('--http_port', type=int, default=8080,
                        help='server port for receiving HTTP requests')
    return parser


def set_service_parser(parser=None):
    from ..service.base import SocketType, BaseService, ParallelType

    import os
    if not parser:
        parser = set_base_parser()

    parser.add_argument('--port_in', type=int, default=random_port(-1),
                        help='port for input data, default a random port between [49152, 65536]')
    parser.add_argument('--port_out', type=int, default=random_port(-1),
                        help='port for output data, default a random port between [49152, 65536]')
    parser.add_argument('--host_in', type=str, default=BaseService.default_host,
                        help='host address for input')
    parser.add_argument('--host_out', type=str, default=BaseService.default_host,
                        help='host address for output')
    parser.add_argument('--socket_in', type=SocketType.from_string, choices=list(SocketType),
                        default=SocketType.PULL_BIND,
                        help='socket type for input port')
    parser.add_argument('--socket_out', type=SocketType.from_string, choices=list(SocketType),
                        default=SocketType.PUSH_BIND,
                        help='socket type for output port')
    parser.add_argument('--port_ctrl', type=int, default=os.environ.get('GNES_CONTROL_PORT', random_port(-1)),
                        help='port for controlling the service, default a random port between [49152, 65536]')
    parser.add_argument('--timeout', type=int, default=-1,
                        help='timeout (ms) of all communication, -1 for waiting forever')
    parser.add_argument('--dump_interval', type=int, default=5,
                        help='serialize the model in the service every n seconds if model changes. '
                             '-1 means --read_only. ')
    parser.add_argument('--read_only', action='store_true', default=False,
                        help='do not allow the service to modify the model, '
                             'dump_interval will be ignored')
    parser.add_argument('--parallel_backend', type=str, choices=['thread', 'process'], default='thread',
                        help='parallel backend of the service')
    parser.add_argument('--num_parallel', '--replicas', type=int, default=1,
                        help='number of parallel services running at the same time (i.e. replicas), '
                             '`port_in` and `port_out` will be set to random, '
                             'and routers will be added automatically when necessary')
    parser.add_argument('--parallel_type', '--replica_type', type=ParallelType.from_string, choices=list(ParallelType),
                        default=ParallelType.PUSH_NONBLOCK,
                        help='parallel type of the concurrent services')
    parser.add_argument('--check_version', action=ActionNoYes, default=True,
                        help='comparing the GNES and proto version of incoming message with local setup, '
                             'mismatch raise an exception')
    parser.add_argument('--identity', type=str, default='',
                        help='identity of the service, empty by default')
    parser.add_argument('--route_table', action=ActionNoYes, default=False,
                        help='showing a route table with time cost after receiving the result')
    parser.add_argument('--squeeze_pb', action=ActionNoYes, default=True,
                        help='sending bytes and ndarray separately apart from the protobuf message, '
                             'usually yields better network efficiency')
    parser.add_argument('--ctrl_with_ipc', action='store_true', default=False,
                        help='use ipc protocol for control socket')
    return parser


def _set_client_parser(parser=None):
    from ..service.base import SocketType
    if not parser:
        parser = set_base_parser()
    set_service_parser(parser)
    parser.set_defaults(
        port_in=parser.get_default('port_out'),
        port_out=parser.get_default('port_in'),
        socket_in=SocketType.PULL_CONNECT,
        socket_out=SocketType.PUSH_CONNECT,
        read_only=True)
    return parser


def _set_loadable_service_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    from ..service.base import SocketType
    set_service_parser(parser)

    parser.add_argument('--yaml_path', type=resolve_yaml_path, required=True,
                        help='yaml config of the service, it should be a readable stream,'
                             ' or a valid file path, or a supported class name.')
    parser.add_argument('--py_path', type=resolve_py_path, nargs='+',
                        help='the file path(s) of an external python module(s).')

    parser.set_defaults(socket_in=SocketType.PULL_BIND,
                        socket_out=SocketType.PUSH_BIND)
    return parser


def _set_sortable_service_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_loadable_service_parser(parser)

    parser.add_argument('--sorted_response', action='store_true', default=False,
                        help='sort the response (if exist) by the score')
    return parser


# shortcut to keep consistent
set_encoder_parser = _set_loadable_service_parser


def set_preprocessor_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_loadable_service_parser(parser)
    parser.set_defaults(read_only=True)
    return parser


def set_healthcheck_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='host address of the checked service')
    parser.add_argument('--port', type=int, required=True,
                        help='control port of the checked service')
    parser.add_argument('--timeout', type=int, default=1000,
                        help='timeout (ms) of one check, -1 for waiting forever')
    parser.add_argument('--retries', type=int, default=3,
                        help='max number of tried health checks before exit 1')
    return parser


def set_router_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_sortable_service_parser(parser)

    parser.add_argument('--num_part', type=int, default=None,
                        help='explicitly set the number of parts of message')
    parser.set_defaults(read_only=True)
    return parser


def set_indexer_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_sortable_service_parser(parser)
    parser.add_argument('--as_response', type=ActionNoYes, default=True,
                        help='convert the message type from request to response after indexing. '
                             'turn it off if you want to chain other services after this index service.')

    return parser


def _set_grpc_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    parser.add_argument('--grpc_host',
                        type=str,
                        default='0.0.0.0',
                        help='host address of the grpc service')
    parser.add_argument('--grpc_port',
                        type=int,
                        default=8800,
                        help='host port of the grpc service')
    parser.add_argument('--max_message_size', type=int, default=-1,
                        help='maximum send and receive size for grpc server in bytes, -1 means unlimited')
    parser.add_argument('--proxy', action=ActionNoYes, default=False,
                        help='respect the http_proxy and https_proxy environment variables. '
                             'otherwise, it will unset these proxy variables before start. '
                             'gRPC seems perfer --no_proxy')
    return parser


def set_grpc_service_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    set_service_parser(parser)
    _set_grpc_parser(parser)
    parser.add_argument('--pb2_path',
                        type=str,
                        required=True,
                        help='the path of the python file protocol buffer compiler')
    parser.add_argument('--pb2_grpc_path',
                        type=str,
                        required=True,
                        help='the path of the python file generated by the gRPC Python protocol compiler plugin')
    parser.add_argument('--stub_name',
                        type=str,
                        required=True,
                        help='the name of the gRPC Stub')
    parser.add_argument('--api_name',
                        type=str,
                        required=True,
                        help='the api name for calling the stub')
    return parser


def set_frontend_parser(parser=None):
    from ..service.base import SocketType
    if not parser:
        parser = set_base_parser()
    set_service_parser(parser)
    _set_grpc_parser(parser)
    parser.set_defaults(socket_in=SocketType.PULL_BIND,
                        socket_out=SocketType.PUSH_BIND,
                        read_only=True)
    parser.add_argument('--max_concurrency', type=int, default=10,
                        help='maximum concurrent connections allowed')
    parser.add_argument('--dump_route', type=argparse.FileType('w', encoding='utf8'),
                        help='dumping route information to a file')
    parser.add_argument('--max_pending_request', type=int, default=100,
                        help='maximum number of pending requests allowed, when exceed wait until we receive the response')
    return parser


def set_client_cli_parser(parser=None):
    import sys
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--txt_file', type=argparse.FileType('r'),
                       default=sys.stdin,
                       help='text file to be used, each line is a doc/query')
    group.add_argument('--image_zip_file', type=str,
                       help='image zip file to be used, consists of multiple images')
    group.add_argument('--video_zip_file', type=str,
                       help='video zip file to be used, consists of multiple videos')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='the size of the request to split')
    parser.add_argument('--mode', choices=['index', 'query', 'train'], type=str,
                        required=True,
                        help='the mode of the client and the server')
    parser.add_argument('--top_k', type=int,
                        default=10,
                        help='top_k results returned in the query mode')
    parser.add_argument('--start_doc_id', type=int,
                        default=0,
                        help='the start number of doc id')
    parser.add_argument('--max_concurrency', type=int, default=10,
                        help='maximum concurrent connections allowed')
    return parser


def set_client_http_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)
    parser.add_argument('--http_port', type=int, default=80,
                        help='http port to deploy the service')
    parser.add_argument('--http_host', type=str, default='0.0.0.0',
                        help='http host to deploy the service')
    parser.add_argument('--max_workers', type=int, default=100,
                        help='max workers to deal with the message')
    parser.add_argument('--top_k', type=int, default=10,
                        help='default top_k for query mode')
    parser.add_argument('--batch_size', type=int, default=2560,
                        help='batch size for feed data for train mode')
    return parser


def get_main_parser():
    # create the top-level parser
    parser = set_base_parser()
    adf = argparse.ArgumentDefaultsHelpFormatter
    sp = parser.add_subparsers(dest='cli', title='GNES sub-commands',
                               description='use "gnes [sub-command] --help" '
                                           'to get detailed information about each sub-command')

    # microservices
    set_frontend_parser(sp.add_parser('frontend', help='start a frontend service', formatter_class=adf))
    set_encoder_parser(sp.add_parser('encode', help='start an encoder service', formatter_class=adf))
    set_indexer_parser(sp.add_parser('index', help='start an indexer service', formatter_class=adf))
    set_router_parser(sp.add_parser('route', help='start a router service', formatter_class=adf))
    set_preprocessor_parser(sp.add_parser('preprocess', help='start a preprocessor service', formatter_class=adf))
    set_grpc_service_parser(sp.add_parser('grpc', help='start a general purpose grpc service', formatter_class=adf))

    pp = sp.add_parser('client', help='start a GNES client of the selected type')
    spp = pp.add_subparsers(dest='client', title='GNES client sub-commands',
                            description='use "gnes client [sub-command] --help" '
                                        'to get detailed information about each client sub-command')
    spp.required = True
    # clients
    set_client_http_parser(
        spp.add_parser('http', help='start a client that allows HTTP requests as input', formatter_class=adf))
    set_client_cli_parser(spp.add_parser('cli', help='start a client that allows stdin as input', formatter_class=adf))

    # others
    set_composer_flask_parser(
        sp.add_parser('compose', help='start a GNES Board to visualize YAML configs', formatter_class=adf))
    set_healthcheck_parser(
        sp.add_parser('healthcheck', help='do health check on any GNES microservice', formatter_class=adf))
    return parser
