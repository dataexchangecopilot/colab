from enum import Enum

class RequestKey(Enum):
    use_case =1
    action = 2
    category = 3
    sentence = 4
    audio_64string = 5
    training_file = 6
    column_name = 7
    training_folder = 8
    predict_file = 9
    question = 10
    output_file = 11
    image_64string = 12
    predict_folder = 13
    company_code = 14
    file_name = 15
    content_64string = 16
    reference_column =17
    text_column = 18
    person_64string = 19
    garment_64string = 20
    video_id = 21
    video_interval= 22
    ocr_sentence = 23
    garment_category = 24
    
class ResponseKey(Enum):
    results = 2
    category = 3
    sentence = 4
    question = 5
    answer = 6
    metadatas = 7
    image_64string = 8

class UseCase(Enum):
    search_by_text = 1
    search_by_voice = 2
    search_image = 3
    remove_image_background = 4
    plm_rag = 5
    plm_ocr = 6
    image_resize = 7
    image_attributes = 8
    voice_to_text = 9
    virtual_try_on = 10
    image_description = 11
    youtube_tutorial = 12

class ActionType(Enum):
    train = 1
    predict = 2
    get_category = 3
    fine_tuning = 4



