# see s_command.py
#from rembg._version import get_versions
import base64
from open_webui.backend_customer.plm_rembg.bg import remove
from open_webui.backend_customer.config import *
from open_webui.backend_customer.plm_enum import RequestKey, UseCase, ResponseKey
class PLMRembg:
    def __init__(self) -> None:
        self._root_directory = ROOT_DIRECTORY

    def do(self, request_paras, company_code):
        if RequestKey.image_64string.name in request_paras:
            image_64string = request_paras[RequestKey.image_64string.name]
            image_bytes = base64.b64decode(image_64string)
        else:
            input_file = request_paras[RequestKey.predict_file.name]
            # format input_file
            input_file = f"{self._root_directory}/{company_code}/{input_file}"
            with open(input_file, "rb") as f:
                image_bytes = f.read()
                
        removed_bytes = remove(image_bytes)
        if RequestKey.output_file.name in request_paras:
            output_file = request_paras[RequestKey.output_file.name]
            output_file = f"{self._root_directory}/{company_code}/{output_file}"
            with open(output_file, "wb") as fw:
                fw.write(removed_bytes)
                
        base64_bytes = base64.b64encode(removed_bytes)
        base64_string = base64_bytes.decode('utf-8')
        return {ResponseKey.image_64string.name: base64_string}