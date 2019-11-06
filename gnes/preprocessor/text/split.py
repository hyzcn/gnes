

import json
import re
import string

from ..base import BaseTextPreprocessor
from ...proto import gnes_pb2


class SentSplitPreprocessor(BaseTextPreprocessor):
    def __init__(self,
                 min_sent_len: int = 1,
                 max_sent_len: int = 256,
                 deliminator: str = '.!?。！？',
                 is_json: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.deliminator = deliminator
        self.is_json = is_json

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        d = doc.raw_bytes.decode()
        if self.is_json:
            d = json.loads(d)
            doc.raw_text = d.pop('Content')
            doc.meta_info = json.dumps(d).encode()
        else:
            doc.raw_text = d

        ret = [(m.group(0), m.start(), m.end()) for m in
               re.finditer(r'[^{0}]+[{0}]'.format(self.deliminator), doc.raw_text)]
        if not ret:
            ret = [(doc.raw_text, 0, len(doc.raw_text))]
        for ci, (r, s, e) in enumerate(ret):
            f = ''.join(filter(lambda x: x in string.printable, r))
            f = re.sub('\n+', ' ', f).strip()
            if len(f) > self.min_sent_len:
                c = doc.chunks.add()
                c.doc_id = doc.doc_id
                c.text = f[:self.max_sent_len]
                c.offset = ci
                c.weight = len(c.text) / len(doc.raw_text)
                c.offset_nd.extend([s, e])
