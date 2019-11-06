


import sys
import time
import zipfile
from typing import Iterator, Tuple

from termcolor import colored

from .base import GrpcClient
from ..proto import RequestGenerator


class CLIClient(GrpcClient):
    def __init__(self, args, start_at_init: bool = True):
        super().__init__(args)
        self._bytes_generator = self._get_bytes_generator_from_args(args)
        if start_at_init:
            self.start()

    @staticmethod
    def _get_bytes_generator_from_args(args):
        if args.txt_file:
            all_bytes = (v.encode() for v in args.txt_file)
        elif args.image_zip_file:
            zipfile_ = zipfile.ZipFile(args.image_zip_file)
            all_bytes = (zipfile_.open(v).read() for v in zipfile_.namelist())
        elif args.video_zip_file:
            zipfile_ = zipfile.ZipFile(args.video_zip_file)
            all_bytes = (zipfile_.open(v).read() for v in zipfile_.namelist())
        else:
            all_bytes = None
        return all_bytes

    def start(self):
        try:
            getattr(self, self.args.mode)()
        except Exception as ex:
            self.logger.error(ex)
        finally:
            self.close()

    def train(self) -> None:
        with ProgressBar(task_name=self.args.mode) as p_bar:
            for _ in self._stub.StreamCall(RequestGenerator.train(self.bytes_generator,
                                                                  doc_id_start=self.args.start_doc_id,
                                                                  batch_size=self.args.batch_size)):
                p_bar.update()

    def index(self) -> None:
        with ProgressBar(task_name=self.args.mode) as p_bar:
            for _ in self._stub.StreamCall(RequestGenerator.index(self.bytes_generator,
                                                                  doc_id_start=self.args.start_doc_id,
                                                                  batch_size=self.args.batch_size)):
                p_bar.update()

    def query(self) -> Iterator[Tuple]:
        for idx, q in enumerate(self.bytes_generator):
            for req in RequestGenerator.query(q, request_id_start=idx, top_k=self.args.top_k):
                resp = self._stub.Call(req)
                yield (req, resp)

    @property
    def bytes_generator(self) -> Iterator[bytes]:
        if self._bytes_generator:
            return self._bytes_generator
        else:
            raise ValueError('bytes_generator is empty or not set')

    @bytes_generator.setter
    def bytes_generator(self, bytes_gen: Iterator[bytes]):
        if self._bytes_generator:
            self.logger.warning('bytes_generator is not empty, overrided')
        self._bytes_generator = bytes_gen


class ProgressBar:
    def __init__(self, bar_len: int = 20, task_name: str = ''):
        self.bar_len = bar_len
        self.task_name = task_name

    def update(self):
        self.num_bars += 1
        sys.stdout.write('\r')
        elapsed = time.perf_counter() - self.start_time
        elapsed_str = colored('elapsed', 'yellow')
        speed_str = colored('speed', 'yellow')
        num_bars = self.num_bars % self.bar_len
        num_bars = self.bar_len if not num_bars and self.num_bars else max(num_bars, 1)

        sys.stdout.write(
            '{:>10} [{:<{}}]  {:>8}: {:3.1f}s   {:>8}: {:3.1f} batch/s'.format(
                colored(self.task_name, 'cyan'),
                colored('=' * num_bars, 'green'),
                self.bar_len + 9,
                elapsed_str,
                elapsed,
                speed_str,
                self.num_bars / elapsed,
            ))
        if num_bars == self.bar_len:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.num_bars = -1
        self.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write('\t%s\n' % colored('done!', 'green'))
