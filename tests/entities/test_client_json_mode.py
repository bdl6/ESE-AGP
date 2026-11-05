"""结构化JSON约束按需启用，保持旧客户端默认协议。"""
import json
from ese_agp.entities.client import ChatClient


def test_json_mode_request():
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,size):return b'{"choices":[]}'
    class Opener:
        def open(self,request,timeout):
            body=json.loads(request.data)
            assert body['response_format']=={'type':'json_object'}
            return Response()
    client=ChatClient({'base_url':'http://127.0.0.1:18888/v1','model':'local','json_mode':True})
    client.opener=Opener()
    assert client.complete('JSON','{}')=={'choices':[]}
