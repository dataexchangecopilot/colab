import easyocr
import base64
from open_webui.backend_customer.plm_enum import RequestKey
class PLMOCR:
    def __init__(self) -> None:
        self._ocr_reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
        pass

    def do(self, request_paras):
        if RequestKey.predict_file.name not in request_paras:
            image_64string = request_paras[RequestKey.image_64string.name]
            image_bytes = base64.b64decode(image_64string)
        else:
            input_file = request_paras[RequestKey.predict_file.name]
            with open(input_file, "rb") as f:
                image_bytes = f.read()
        reader = self._ocr_reader
        result = reader.readtext(image_bytes, detail=0)
        return result