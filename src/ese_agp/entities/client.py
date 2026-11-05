"""OpenAI兼容Chat Completions协议适配；不绑定供应商或模型，不记录密钥。"""
import json
import os
import urllib.request
from urllib.parse import urlparse


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """拒绝重定向，避免Authorization被发送到另一个服务。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """由调用方按HTTP状态处理重定向失败。"""
        return None


class ChatClient:
    """服务地址须包含版本前缀；使用 /chat/completions 和标准消息格式。"""
    def __init__(self, config):
        """密钥仅从指定环境变量读取；初始化不发请求。"""
        self.config=config
        base=config['base_url'].rstrip('/')
        url=urlparse(base)
        if url.scheme!='https' and not (url.scheme=='http' and url.hostname in {'localhost','127.0.0.1','::1'}):
            raise ValueError('Use HTTPS, or localhost HTTP for a local model')
        if url.username or url.password or url.query or url.fragment:
            raise ValueError('Credentials/query not allowed in base_url')
        self.endpoint=base+'/chat/completions'
        self.key=os.environ.get(config.get('api_key_env','ESE_ENTITY_API_KEY'),'')
        if not self.key and url.hostname not in {'localhost','127.0.0.1','::1'}:
            raise ValueError('Missing API key environment variable: '+config.get('api_key_env','ESE_ENTITY_API_KEY'))
        self.opener=urllib.request.build_opener(NoRedirect)

    def complete(self, prompt, text):
        """返回原始JSON；超时、HTTP错误交给执行器分类重试。"""
        body={'model':self.config['model'],'messages':[{'role':'system','content':prompt},
              {'role':'user','content':text}],'temperature':0,'max_tokens':self.config.get('max_output_tokens',1024)}
        if self.config.get('json_mode',False):body['response_format']={'type':'json_object'}
        request=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+self.key} if self.key else {})},method='POST')
        with self.opener.open(request,timeout=self.config.get('timeout_seconds',60)) as response:
            return json.loads(response.read(4*1024*1024))
