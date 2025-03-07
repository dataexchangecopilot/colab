# https://github.com/jdepoix/youtube-transcript-api
import torch
import os, time, base64
import yt_dlp
import clip
import cv2
import easyocr
from io import BytesIO
from PIL import Image, ImageFile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from open_webui.backend_customer.plm_enum import RequestKey, ActionType
from open_webui.backend_customer.config import *
from langchain_ollama import ChatOllama
from yt_dlp.utils import download_range_func
class Prompt:
    @staticmethod
    def prompt1(ID=0):
        if ID == 0:
            prompt_text = """Your task: Condense a video transcript into a captivating and informative 250-word summary that highlights key points and engages viewers.

Guidelines:
    Focus on essential information: Prioritize the video's core messages, condensing them into point-wise sections.
    Maintain clarity and conciseness: Craft your summary using accessible language, ensuring it's easily understood by a broad audience.
    Capture the essence of the video: Go beyond mere listings. Integrate key insights and interesting aspects to create a narrative that draws readers in.
    Word count: Aim for a maximum of 250 words.

Input:
    The provided video transcript will be your content source.

Example (for illustration purposes only):
    Setting the Stage: Briefly introduce the video's topic and context.
    Key Points:
        Point A: Describe the first crucial aspect with clarity and depth.
        Point B: Elaborate on a second significant point.
        (Continue listing and describing key points)
    Conclusions: Summarize the video's main takeaways, leaving readers with a clear understanding and potential interest in learning more.

Additional Tips:
    Tailor the tone: Adjust your language to resonate with the video's intended audience and overall style.
    Weave in storytelling elements: Employ vivid descriptions and engaging transitions to make the summary more memorable.
    Proofread carefully: Ensure your final summary is free of grammatical errors and typos.

By following these guidelines, you can effectively transform video transcripts into captivating and informative summaries, drawing in readers and conveying the video's essence effectively."""

        elif ID == "timestamp":
            prompt_text = """
            Generate timestamps for main chapter/topics in a YouTube video transcript.
            Given text segments with their time, generate timestamps for main topics discussed in the video. Format timestamps as hh:mm:ss and provide clear and concise topic titles.  
           
            Instructions:
            1. List only topic titles and timestamps.
            2. Do not explain the titles.
            3. Include clickable URLs.
            4. Provide output in Markdown format.

            Markdown for output:
            1. [hh:mm:ss](%VIDEO_URL?t=seconds) %TOPIC TITLE 1%
            2. [hh:mm:ss](%VIDEO_URL?t=seconds) %TOPIC TITLE 2%
            - and so on

            Markdown Example:
            1. [00:05:23](https://youtu.be/hCaXor?t=323) Introduction
            2. [00:10:45](https://youtu.be/hCaXor?t=645) Main Topic 1
            3. [00:25:17](https://youtu.be/hCaXor?t=1517) Main Topic 2
            - and so on

            The %VIDEO_URL% (YouTube video link) and transcript are provided below :
            """
            
        elif ID == "transcript":
            prompt_text = """
            """

        else:
            prompt_text = """
            Can you summarize the main points of this video (or transcript)? 
            Please include the key takeaways with time stamps, 
            formatted with headings and bullet points""" 

        return prompt_text

class PLMYoutubeTutorial:
    def __init__(self) -> None:
        self._score_column = "score"
        self._db_directory = DB_DIRECTORY
        self._root_directory = ROOT_DIRECTORY
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self._device)
        self._vectorizer = model
        self._preprocess = preprocess
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    def _get_video_url(self, video_id):
        return  f"https://www.youtube.com/watch?v={video_id}"
    def _get_int(self, filename):
        name_without_extension = filename.split('.')[0]
        return int(name_without_extension) 
    def _parse_no_chapter(self, transcript, title, video_id):
        video_url = self._get_video_url(video_id)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0
        )
        # llm = ChatOllama(
        #     model="llama3.1",
        #     temperature=0,
        #     format="json"
        # )
        prompt = Prompt.prompt1(ID='timestamp')
        messages = [SystemMessage(content= prompt + " VIDEO_URL:" + video_url),
            HumanMessage(content=str(transcript))
        ]
        ai_msg = llm.invoke(messages)
        video_metadata ={
            "title":title,
            "start_time":0,
            "end_time":-1,
            "content":ai_msg.content,
            "youtube_url":video_url
        }
        #self._download_video(video_id, 1)
        return [video_metadata]
    
    def _parse_chapters(self, transcript,chapters, video_id):
        video_url = self._get_video_url(video_id)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0
        )
        # llm = ChatOllama(
        #     model="llama3.1",
        #     temperature=0,
        #     format="json"
        # )
        video_metadatas =[]

        for video_chapter in chapters:
            # filter transcription by chapter
            # start_time <= start <= end_time 
            title = video_chapter["title"]
            start_time = video_chapter["start_time"]
            end_time  =video_chapter["end_time"]
            filtered_transcript = [item for item in transcript if  start_time <= item["start"] <= end_time]
            print(filtered_transcript)
            print("")
            prompt = Prompt.prompt1(ID='timestamp')
            messages = [SystemMessage(content= prompt + " VIDEO_URL:" + video_url),
                HumanMessage(content=str(filtered_transcript))
            ]
            ai_msg = llm.invoke(messages)
            video_metadata ={
                "title":title,
                "start_time":start_time,
                "end_time":end_time,
                "content":ai_msg.content,
                "youtube_url":f"{video_url}&t={start_time}"
            }
            video_metadatas.append(video_metadata)
            time.sleep(10)
            #self._download_video(video_id, start_time+1)
        return video_metadatas

    def _save_to_db(self, metadatas,video_id, collection_name):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name = model_name)
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._db_directory
        )
        texts = [f"{c['title']} \n {c['content']}" for c in metadatas]
        ids= [f"{video_id}_{c['start_time']}" for c in metadatas]
        vectors = embeddings.embed_documents(texts)
        db._collection.upsert(ids= ids, embeddings= vectors, metadatas=metadatas, documents=texts)
    
    def _search_by_title(self, search_text,collection_name):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name = model_name)
        db = Chroma(
            collection_name=collection_name,
            persist_directory= self._db_directory,
            embedding_function=embeddings
        )
        # 0 is dissimilar, 1 is most similar.
        docs_with_score = db.similarity_search_with_relevance_scores(
            search_text
        )
        
        results =[]
        for doc, score in docs_with_score:
            if score > 0:
                result = doc.metadata
                result["score"] = "{:.4f}".format(score)
                results.append(result)
        return results

    def _get_image(self, image_path):
        image = self._preprocess(Image.open(image_path))
        img_data = image.unsqueeze(0).to(self._device)
        return img_data
    
    def _get_image_ocr(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        cuda = torch.cuda.is_available()
        reader = easyocr.Reader(['en'], gpu=cuda)
        result= reader.readtext(image_bytes, detail=0)
        return " ".join(result)
    
    def _get_image_ocr_by_bytes(self, image_bytes):
        cuda = torch.cuda.is_available()
        reader = easyocr.Reader(['en'], gpu=cuda)
        result= reader.readtext(image_bytes, detail=0)
        return " ".join(result)
    
    # get screenshot from video_id folder
    # file name shoud be sorted by chapter    
    def _save_thumbnails(self, metadatas, video_id,collection_name):        
        images = []
        image_names = []
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._db_directory
        )
        thumbnail_folder = f"{self._root_directory}/{video_id}"
        for image_name in os.listdir(thumbnail_folder):
            image_path = os.path.join(thumbnail_folder, image_name)
            images.append(self._get_image(image_path))
            image_names.append(f"{video_id}_{image_name}")
        image_inputs = torch.cat(images)
        image_vectors = self._vectorizer.encode_image(image_inputs)
        # Save the image_vectors and image_names to database
        db._collection.upsert(ids= image_names, embeddings=image_vectors.tolist(), metadatas=metadatas,documents=image_names)
    
    def _save_frames(self, video_id,collection_name):        
        images = []
        image_names = []
        metadatas =[]
        video_url = self._get_video_url(video_id)
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._db_directory
        )
        thumbnail_folder = f"{self._root_directory}/{video_id}"
        for image_name in os.listdir(thumbnail_folder):
            image_path = os.path.join(thumbnail_folder, image_name)
            images.append(self._get_image(image_path))
            image_names.append(f"{video_id}_{image_name}")
            start_time = self._get_int(image_name)
            metadata ={"youtube_url":f"{video_url}&t={start_time}" }
            metadatas.append(metadata)
        image_inputs = torch.cat(images)
        image_vectors = self._vectorizer.encode_image(image_inputs)
        # Save the image_vectors and image_names to database
        db._collection.upsert(ids= image_names, embeddings=image_vectors.tolist(), metadatas=metadatas, documents=image_names)
        
    def _save_frame_ocrs(self, video_id,collection_name):        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name = model_name)
        image_ocrs = []
        image_names = []
        metadatas =[]
        video_url = self._get_video_url(video_id)
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._db_directory
        )
        thumbnail_folder = f"{self._root_directory}/{video_id}"
        for image_name in os.listdir(thumbnail_folder):
            image_path = os.path.join(thumbnail_folder, image_name)
            image_ocr = self._get_image_ocr(image_path)
            image_ocrs.append(image_ocr)
            image_names.append(f"{video_id}_{image_name}")
            start_time = self._get_int(image_name)
            metadata ={
                "youtube_url":f"{video_url}&t={start_time}", 
                "image_ocr":image_ocr 
            }
            metadatas.append(metadata)
        vectors = embeddings.embed_documents(image_ocrs)
        # Save the image_vectors and image_names to database
        db._collection.upsert(ids= image_names, embeddings=vectors, metadatas=metadatas, documents=image_names)
        
    def _search_by_thumbnail(self, image_path, collection_name):        
        image = self._get_image(image_path)
        image_vector = self._vectorizer.encode_image(image)
        db = Chroma(
            collection_name= collection_name, 
            persist_directory= self._db_directory
        )
        docs_with_score = db.similarity_search_by_vector_with_relevance_scores(embedding=image_vector.tolist())
        results =[]
        for doc, score in docs_with_score:
            if score < 1000:
                result = doc.metadata
                result["score"] = str(f"{1 - score:.4f}")
                results.append(result)
        return results
    
    def _get_youtube(self, youtube_id, collection_name, video_interval):
        transcript = YouTubeTranscriptApi.get_transcript(youtube_id)
        video_url = self._get_video_url(youtube_id)
        ydl_opts = {}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', None)
            video_chapters = info_dict.get('chapters', None)
        if video_chapters == None:
            data = self._parse_no_chapter(transcript, video_title, youtube_id) 
        else:
            data = self._parse_chapters(transcript, video_chapters, youtube_id)
        self._save_to_db(data,youtube_id, collection_name)
        thumbnail_collection = f"{collection_name}_thumbnail"
        #self._save_thumbnails(data, youtube_id,thumbnail_collection)
        self._download_video_frame(youtube_id, video_interval)
        self._save_frames(youtube_id,thumbnail_collection)
        ocr_collection = f"{collection_name}_thumbnail_ocr"
        self._save_frame_ocrs(youtube_id,ocr_collection)
        
    def do(self, request_paras, company_code):
        action_type = request_paras[RequestKey.action.name]
        if len(company_code) > 0:
            collection_name = f"{company_code}_youtube"
        else:
            collection_name = "youtube"
            
        if (action_type == ActionType.train.name):
            video_id = request_paras[RequestKey.video_id.name]
            video_interval = request_paras[RequestKey.video_interval.name]
            self._get_youtube(video_id,collection_name, video_interval)
            return_data= "train is done."
        else:
            if RequestKey.image_64string.name in request_paras:
                image_64string = request_paras[RequestKey.image_64string.name]
                image_bytes = base64.b64decode(image_64string)
                image_path = BytesIO(image_bytes)
                image_collection = f"{collection_name}_thumbnail"
                return_data = self._search_by_thumbnail(image_path, image_collection)
                ocr_collection = f"{collection_name}_thumbnail_ocr"
                sentence = self._get_image_ocr_by_bytes(image_bytes) 
                return_data.extend(self._search_by_title(sentence, ocr_collection))
            elif RequestKey.predict_file.name in request_paras:
                image_path = request_paras[RequestKey.predict_file.name]
                image_collection = f"{collection_name}_thumbnail"
                return_data = self._search_by_thumbnail(image_path, image_collection)
            elif RequestKey.ocr_sentence.name in request_paras:
                sentence = request_paras[RequestKey.ocr_sentence.name]
                ocr_collection = f"{collection_name}_thumbnail_ocr"
                return_data = self._search_by_title(sentence, ocr_collection)
            else:
                sentence = request_paras[RequestKey.sentence.name]
                return_data = self._search_by_title(sentence, collection_name)
        return return_data                
# prompt
# Summarize the following video transcript. 
# Include all the key points and 
# format your summary with headings and bullet points to make it quickly skimmable.
# Can you summarize the main points of this video (or transcript)? Please include the key takeaways with time stamps, formatted with headings and bullet points

# I have a long transcription to summarize. Here's the first part (insert segment here). Please summarize the key points with time stamps, and let me know when you're ready for the next segment.
    def _download_video(self, video_id, start_second):
        # URL of the YouTube video
        video_url = self._get_video_url(video_id)
        # Define the options for yt-dlp
        # Define a custom postprocessor to capture the output file name
        class CustomPostProcessor(yt_dlp.postprocessor.common.PostProcessor):
            def __init__(self):
                super().__init__()
                self.output_file = None

            def run(self, info):
                self.output_file = info['filepath']
                return [], info
        ydl_opts = {
            'download_ranges': download_range_func(None, [(start_second, start_second + 1)]),
            'outtmpl': '%(timestamp)s.%(ext)s',  # Output file name template
            'format': 'bestvideo+bestaudio',  # Best quality video and audio
        }
        customPostProcessor = CustomPostProcessor()
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.add_post_processor(customPostProcessor)
            result = ydl.download([video_url])
            output_file = customPostProcessor.output_file
            
        cap = cv2.VideoCapture(output_file)
        #fps = cap.get(cv2.CAP_PROP_FPS)    
        #frame_number = int(fps * timestamp)
        #set = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        cap.release()
        os.remove(output_file)
        image_folder = f"{self._root_directory}/{video_id}"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        str_start = f'{int(start_second):08}'
        temp_file = f"{self._root_directory}/{video_id}/{str_start}.jpg"
        print(temp_file)
        if success:
            # Save the frame to the temporary file
            cv2.imwrite(temp_file, frame)
        else:
            print("failed capture frame")

    def _download_video_frame(self, video_id,inter_second):
        # URL of the YouTube video
        video_url = self._get_video_url(video_id)
        # Define the options for yt-dlp
        # Define a custom postprocessor to capture the output file name
        class CustomPostProcessor(yt_dlp.postprocessor.common.PostProcessor):
            def __init__(self):
                super().__init__()
                self.output_file = None

            def run(self, info):
                self.output_file = info['filepath']
                return [], info
        ydl_opts = {
            'outtmpl': '%(timestamp)s.%(ext)s',  # Output file name template
            'format': 'bestvideo+bestaudio',  # Best quality video and audio
        }
        customPostProcessor = CustomPostProcessor()
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.add_post_processor(customPostProcessor)
            result = ydl.download([video_url])
            output_file = customPostProcessor.output_file
        # create download folder    
        image_folder = f"{self._root_directory}/{video_id}"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        cap = cv2.VideoCapture(output_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_second = 1
        while True:
            frame_number = int(fps * start_second)
            set = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
            if success:
            # Save the frame to the temporary file
                temp_file = f"{image_folder}/{start_second:08}.jpg"
                cv2.imwrite(temp_file, frame)
                start_second = start_second + inter_second
            else:
                break
        cap.release()
        os.remove(output_file)


        
        


