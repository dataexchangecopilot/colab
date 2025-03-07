import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .modules.plm_search_text_similar_chroma import PLMSearchText
from .plm_rembg.plm_rembg import PLMRembg
from .modules.plm_ocr import PLMOCR
from .modules.plm_search_image_clip import PLMSearchImage
from .modules.plm_get_transcripts_from_youtube_chroma import PLMYoutubeTutorial
from .tryon.plm_tryon import PLMTryon
from .utility import Utility
from .plm_enum import RequestKey, ResponseKey, UseCase, ActionType
from .config import *

router = APIRouter()

image_instance = PLMSearchImage()
text_instance = PLMSearchText()
rembg_instance = PLMRembg()
ocr_instance = PLMOCR()
youtube_tutorial_instance = PLMYoutubeTutorial()
tryon_instance = PLMTryon()
utility_instance = Utility()

@router.post('')
async def post_plm(request: Request):
    req_json = await request.json()
    return_data =""
    try:
        use_case = req_json[RequestKey.use_case.name]
        company_code = utility_instance.get_company_code(req_json)
        # search image  --- repalced by ML .net
        if use_case == UseCase.search_image.name:
            return image_instance.do(req_json, company_code)
        # search plm search view --- Search, menu item action ,image /file uplaod 
        elif use_case == UseCase.search_by_text.name:
            action_type = req_json[RequestKey.action.name]
            if (action_type == ActionType.predict.name):
                if RequestKey.category.name not in req_json:
                    sentence = req_json[RequestKey.sentence.name]
                    #category_datas = text_instance.get_category_datas("category")
                    #category = utility_instance.parse_category(sentence,category_datas)
                    category = utility_instance.parse_category(sentence)
                    req_json[RequestKey.category.name] = category
            return text_instance.do(req_json, company_code)
        # plm rag -- knowlege base seach 
        # elif use_case == UseCase.plm_rag.name:
        #     return rag_instance.do(req_json,company_code)
        # remove image's background 
        elif use_case == UseCase.remove_image_background.name:
            return rembg_instance.do(req_json, company_code)
        # search by voice
        elif use_case == UseCase.search_by_voice.name:
            audio_64string = req_json[RequestKey.audio_64string.name]
            audio_text, category = utility_instance.parse_voice(audio_64string)
            if len(category) > 0:
                req_json[RequestKey.category.name] = category
            if len(audio_text) > 0:
                req_json[RequestKey.sentence.name] = audio_text
            return text_instance.do(req_json, company_code)
        elif use_case == UseCase.plm_ocr.name:
            return ocr_instance.do(req_json)
        # elif use_case == UseCase.image_resize.name:
        #     return image_resize_instance.do(req_json)
        # elif use_case == UseCase.image_attributes.name:
        #     return image_attributes_instance.do(req_json)
        elif use_case == UseCase.voice_to_text.name:
            audio_64string = req_json[RequestKey.audio_64string.name]
            audio_text = utility_instance.audio_2_text(audio_64string)
            return { ResponseKey.sentence.name:audio_text }  
        elif use_case == UseCase.image_description.name:
            return utility_instance.image_description(req_json)
        elif use_case == UseCase.virtual_try_on.name:
            return tryon_instance.do(req_json)
        elif use_case == UseCase.youtube_tutorial.name:
            return youtube_tutorial_instance.do(req_json, company_code)
        else:
            raise Exception("no that use case")
    except Exception as e:
        return_data = str(e)
        logging.info(return_data)
        raise HTTPException(status_code=500, detail=return_data)
