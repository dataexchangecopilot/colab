# utility 
# https://python.langchain.com.cn/docs/use_cases/chatbots/voice_assistant
# https://github.com/meirm/jarvis/blob/main/jarvis.py
# https://pypi.org/project/SpeechRecognition/
# https://github.com/Uberi/speech_recognition/tree/master
# https://github.com/kkroening/ffmpeg-python
import speech_recognition as sr
import os, io, zipfile, base64, uuid
import ffmpeg
import ollama
from openai import OpenAI
from PIL.Image import Image as PILImage
from google.cloud.speech_v1 import SpeechClient
from google.cloud.speech_v1.types import cloud_speech, RecognitionConfig, RecognitionAudio
from gradio_client import Client, file
from .config import *
from .plm_enum import RequestKey, ResponseKey
#import pyttsx3

class Utility:
    def __init__(self) -> None:
        self._r = sr.Recognizer()
        if len(OPENAI_API_KEY) > 0 :
            self._use_openai = True
        else:
            self._use_openai = False
        if len(GOOGLE_APPLICATION_CREDENTIALS) > 0 :
            self._use_google = True
        else:
            self._use_google = False
        self._root_directory = ROOT_DIRECTORY

    def _audio_2_text_by_speech_recognition(self, audio_data):
        bytes_io = io.BytesIO(audio_data)
        r = self._r
        with sr.AudioFile(bytes_io) as source:
            audio = r.record(source)  # read the entire audio file

        try:
        # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
        # other speech recognition models are also available.
            text = r.recognize_whisper(
                audio,
                model="medium.en",
                show_dict=True,
            )["text"]
        except Exception as e:
            unrecognized_speech_text = (
                f"Sorry, I didn't catch that. Exception was: {e}s"
            )
            text = unrecognized_speech_text
        print(text)
        return text
    
    def _convert_2_wav_by_ffmpeg_python(self, decoded_bytes):
        # Convert WebM audio to WAV using ffmpeg
        wav_data, _ = (
            ffmpeg
            .input('pipe:')
            .output('pipe:', format='wav')
            .run(input=decoded_bytes, capture_stdout=True)
        )
        return wav_data
    
    def _audio_2_text_by_openai_api(self, decoded_bytes):
        bytes_io = io.BytesIO(decoded_bytes)
        bytes_io.name ="upload.webm"
        client = OpenAI()
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=bytes_io
        )
        return transcription.text
    
    def _audio_2_text_by_google_api(self, decoded_bytes):
        # if use v2 version you check service agents
        # https://cloud.google.com/iam/docs/service-agents
        # https://stackoverflow.com/questions/77598023/speech-to-text-api-v2-issue-with-python-permission-denied
        client = SpeechClient()
        config = RecognitionConfig(language_code ="en")
        audio = RecognitionAudio(content = decoded_bytes)
        request = cloud_speech.RecognizeRequest(
            config = config,
            audio = audio
        )
        response = client.recognize(request=request)
        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")

        return response.results[0].alternatives[0].transcript
    
    def _remove_noise_from_text(self, audio_text):
        noise_list = [",", "!", "."]
        audio_list = audio_text.split(" ")
        string_list =[]
        for item in audio_list:
            if any(char.isdigit() for char in item):
                # If contains digit, remove noise
                for noise in noise_list:
                    item = item.replace(noise, "")
            string_list.append(item)
        separator = " "
        return separator.join(string_list)
     
    def audio_2_text(self, audio_64string):
        audio_bytes = base64.b64decode(audio_64string)
        wav_data = self._convert_2_wav_by_ffmpeg_python(audio_bytes)
        if (self._use_openai):
            audio_text = self._audio_2_text_by_openai_api(wav_data)
        elif self._use_google:
            audio_text = self._audio_2_text_by_google_api(wav_data)     
        else:
            audio_text = self._audio_2_text_by_speech_recognition(wav_data)
        return audio_text.strip().lower()
        
    def image_resize(self, image :PILImage, width:int, height:int):
        return image.resize((width, height))
    
    def unzip_file(self, zip_file_name:str, extracted_folder=None):
        with zipfile.ZipFile(zip_file_name, 'r') as zipf:
        # Extract all files from the zip file
            zipf.extractall(path= extracted_folder)

    def zip_file(self, zip_file_name:str, file_folder=None):
        with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        # Add files to the zip file
        # list files in folders, add one by one
            for file_name in os.listdir(file_folder):
                file_path = os.path.join(file_folder, file_name)
                zipf.write(file_path)
    
    def parse_category(self, long_text):
        product_keys = ["product","reference","open"]
        search_keys =["search","show"]
        main_menu_keys =["go"]
        category ="search"
        if any(key in long_text for key in search_keys):
            category ="search"
        elif any(key in long_text for key in product_keys):
            category ="product"
        elif any(key in long_text for key in main_menu_keys):
            category ="main_menu"
        return category
    
    def parse_voice(self, audio_64string):
        audio_text = self.audio_2_text(audio_64string)
        audio_text = self._remove_noise_from_text(audio_text)
        category ="search"
        if len(audio_text) > 0:
            category = self.parse_category(audio_text)         
        return audio_text, category
    
    def get_company_code(self, request_paras):
        if RequestKey.company_code.name in request_paras:
            company_code = request_paras[RequestKey.company_code.name]
        else:
            company_code = "company_code"
        company_dir = f"{self._root_directory}/{company_code}"
        if not os.path.isdir(company_dir):
            os.makedirs(company_dir)
        return company_code
    
    def image_description(self, request_paras):
        if RequestKey.image_64string.name in request_paras:
            image_64string = request_paras[RequestKey.image_64string.name]
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': 'Can you describe the clothing in this image? Please include details about the style, color, fabric, and any notable features.',
                'images': [image_64string]
            }]
        )
        return response["message"]["content"]

    # https://huggingface.co/spaces/rlawjdghek/StableVITON
    def image_tryon(self, request_paras):
        person_file =  str(uuid.uuid4())+".jpg"
        garment_file =  str(uuid.uuid4())+".jpg"
        if RequestKey.person_64string.name in request_paras:
            person_64string = request_paras[RequestKey.person_64string.name]
            person_bytes = base64.b64decode(person_64string)
            with open(person_file, "wb") as temp_file:
                temp_file.write(person_bytes)
        if RequestKey.garment_64string.name in request_paras:
            garment_64string = request_paras[RequestKey.garment_64string.name]
            garment_bytes = base64.b64decode(garment_64string)
            with open(garment_file, "wb") as temp_file:
                temp_file.write(garment_bytes)
        # client = Client("rlawjdghek/StableVITON")
        # result = client.predict(
        #     vton_img=file(person_file),
        #     garm_img=file(garment_file),
        #     n_steps=20,
        #     is_custom=False,
        #     api_name="/process_hd"
        # )
        # output_file = result
                
        # https://huggingface.co/spaces/AI-Platform/Virtual-Try-On    
        # client = Client("AI-Platform/Virtual-Try-On")
        # result = client.predict(
		#     dict={"background":file(person_file),"layers":[],"composite":None},
		#     garm_img=file(garment_file),
		#     garment_des="Hello!!",
		#     is_checked=True,
		#     is_checked_crop=False,
		#     denoise_steps=30,
		#     seed=42,
		#     api_name="/tryon"
        # )
        # output_file = result[0]
        
        # https://huggingface.co/spaces/paroksh-mason/Virtual-Try-On
        client = Client("paroksh-mason/Virtual-Try-On")
        result = client.predict(
            human_img_dict=file(person_file),
            garm_img=file(garment_file),
            garment_des="",
            background_img=None,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )
        output_file = result[0]
        
        with open(output_file, "rb") as f:
            image_bytes = f.read()
            base64_bytes = base64.b64encode(image_bytes)
            base64_string = base64_bytes.decode('utf-8')
        os.remove(person_file)
        os.remove(garment_file)
        print(output_file)
        return {ResponseKey.image_64string.name: base64_string}
    