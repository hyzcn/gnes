

from .base import YamlComposer, parse_http_data
from ..cli.parser import set_composer_parser
from ..helper import set_logger


class YamlComposerFlask:
    def __init__(self, args):
        self.args = args
        self.logger = set_logger(self.__class__.__name__, self.args.verbose)

    def _create_flask_app(self):
        try:
            from flask import Flask, request
        except ImportError:
            raise ImportError('Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.'
                              'Please use "pip install gnes[flask]" to install it.')

        app = Flask(__name__)
        args = set_composer_parser().parse_args([])
        default_html = YamlComposer(args).build_all()['html']

        @app.errorhandler(500)
        def exception_handler(error):
            self.logger.error('unhandled error, i better quit and restart! %s' % error)
            return '<h1>500 Internal Error</h1> ' \
                   'While we are fixing the issue, do you know you can deploy GNES board locally on your machine? ' \
                   'Simply run <pre>docker run -d -p 0.0.0.0:80:8080/tcp gnes/gnes compose --flask</pre>', 500

        @app.route('/', methods=['GET'])
        def _get_homepage():
            return default_html

        @app.route('/generate', methods=['POST'])
        def _regenerate():
            data = request.form if request.form else request.json
            return parse_http_data(data, args)

        return app

    def run(self):
        app = self._create_flask_app()
        app.run(port=self.args.http_port, threaded=True, host='0.0.0.0')
