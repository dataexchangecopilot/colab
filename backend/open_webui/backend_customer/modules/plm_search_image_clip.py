import os, os.path, base64
import pandas as pd
from io import BytesIO
from langchain_chroma import Chroma
from PIL import Image, ImageFile
from fashion_clip.fashion_clip import FashionCLIP
from open_webui.backend_customer.plm_enum import RequestKey, ActionType
from open_webui.backend_customer.config import *

class image_title_dataset():
    def __init__(self, image_paths, texts):
        self.image_paths = image_paths
        self.texts = texts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        text = self.texts[idx]
        return image, text
    
from transformers import CLIPProcessor, CLIPModel
from fashion_clip.fashion_clip import FashionCLIP, _MODELS
from fashion_clip.utils import _model_processor_hash
from typing import Union
import torch
class PLMCLIP(FashionCLIP):
    def _load_model(self,
                    name: str,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    auth_token = None):
        # model is one of know HF models
        if os.path.isdir(name):
            # if from fine-tuning
            name = _MODELS[name] if name in _MODELS else name
            model = CLIPModel.from_pretrained(name)
            preprocessing = CLIPProcessor.from_pretrained(name)
            hash = _model_processor_hash(name, model, preprocessing)
        else:
            model, preprocessing, hash = super()._load_model(name, device, auth_token)
        return model, preprocessing, hash
    
class PLMSearchImage:
    """ clas for plm search image """
    def __init__(self) -> None:
        self._score_column = "score"
        self._persist_directory = DB_DIRECTORY
        self._root_directory = ROOT_DIRECTORY
        self._num_epochs = 4 
        self._vectorizer = FashionCLIP('fashion-clip')
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    # delete collection
    def _delete_collection(self, collection_name):
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._persist_directory
        )
        db.delete_collection()

    # Compute image vectors for all images in the folder
    def _train(self, request_paras, collection_name, company_code):
        # image_folder
        image_folder = request_paras[RequestKey.training_folder.name]
        image_folder = f"{self._root_directory}/{company_code}/{image_folder}"
        # save train_file
        file_name = request_paras[RequestKey.file_name.name]
        train_file = f"{self._root_directory}/{company_code}/{file_name}"
        content_64string = request_paras[RequestKey.content_64string.name]   
        content_bytes = base64.b64decode(content_64string)
        with open(train_file, "wb") as fw:
            fw.write(content_bytes)
            
        _, file_extension = os.path.splitext(train_file)
        if file_extension.startswith(".xls"):
            df = pd.read_excel(train_file)
        else:
            df = pd.read_csv(train_file)
        # get data from file
        text_column = request_paras[RequestKey.text_column.name]
        image_column = request_paras[RequestKey.reference_column.name]
        # drop items that have the same description
        df = df.drop_duplicates(text_column).copy()
        # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 word
        df = df[df[text_column].apply(lambda x : 0 < len(str(x).split()) < 40)]

        db = Chroma(collection_name= collection_name, 
            persist_directory= self._persist_directory
        )
        text_collection = collection_name+"_text"
        db_text = Chroma(collection_name= text_collection, 
            persist_directory= self._persist_directory
        )
        model = self._get_vectorizer(company_code)

        image_paths = []   
        image_texts = []
        image_names = []
        image_metadatas = []
        for index, row in df.iterrows():
            image_name = str(row[image_column]) + ".jpg"
            image_path = os.path.join(image_folder, image_name)
            image_title = str(row[text_column])
            if os.path.exists(image_path):
                image_paths.append(image_path)
                image_texts.append(image_title)
                image_names.append(image_name)
                image_metadatas.append({"file_name": image_name})
            if (len(image_names) == 512):
                image_vectors = model.encode_images(image_paths, batch_size=128)
                text_vectors = model.encode_text(image_texts, batch_size=128)
                # Save the image_vectors and image_names to database
                db._collection.upsert(ids= image_names, embeddings=image_vectors.tolist(), metadatas=image_metadatas,documents=image_names)
                db_text._collection.upsert(ids= image_names, embeddings=text_vectors.tolist(), metadatas=image_metadatas,documents=image_names)
                image_paths = []
                image_texts = []
                image_names = []
                image_metadatas = []
        image_vectors = model.encode_images(image_paths, batch_size=128)
        text_vectors = model.encode_text(image_texts, batch_size=128)
        # Save the image_vectors and image_names to database
        db._collection.upsert(ids= image_names, embeddings=image_vectors.tolist(), metadatas=image_metadatas,documents=image_names)
        db_text._collection.upsert(ids= image_names, embeddings=text_vectors.tolist(), metadatas=image_metadatas,documents=image_names)
    
    def _find_similar(self, vector, collection_name):
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._persist_directory
        )
        docs_with_score = db.similarity_search_by_vector_with_relevance_scores(embedding=vector.tolist(), k=5)

        results =[]
        for doc, score in docs_with_score:
            result = doc.metadata
            result[self._score_column] = str(f"{1 - score:.4f}")
            #print(result)
            results.append(result)
        return results
    
        #similar_images = [{'name': self._image_names[index], 'similar': str(1 - dist)} for index, dist in zip(indices[0], distances[0])]
        # Return the sorted list
    def _find_by_image(self, image_path, collection_name, company_code):
        model = self._get_vectorizer(company_code)
        image = Image.open(image_path).convert("RGB")
        image = [image]
        image_vector = model.encode_images(image,batch_size=1)
        return self._find_similar(image_vector, collection_name)
    
    def _find_by_text(self, query_text, collection_name, company_code):
        model = self._get_vectorizer(company_code)
        text = [query_text]
        text_vector = model.encode_text(text,batch_size=1)
        result = self._find_similar(text_vector, collection_name)
        text_collection = collection_name+"_text"
        result2 = self._find_similar(text_vector, text_collection)
        # unio two dataset, need to test intersect as well
        result.extend(result2)
        return result
    
    def _build_data(self, request_paras, company_code):
        # image_folder
        image_folder = request_paras[RequestKey.training_folder.name]
        image_folder = f"{self._root_directory}/{company_code}/{image_folder}"
        # save train_file
        file_name = request_paras[RequestKey.file_name.name]
        train_file = f"{self._root_directory}/{company_code}/{file_name}"
        content_64string = request_paras[RequestKey.content_64string.name]   
        content_bytes = base64.b64decode(content_64string)
        with open(train_file, "wb") as fw:
            fw.write(content_bytes)
            
        _, file_extension = os.path.splitext(train_file)
        if file_extension.startswith(".xls"):
            df = pd.read_excel(train_file)
        else:
            df = pd.read_csv(train_file)
        # get data from file
        text_column = request_paras[RequestKey.text_column.name]
        image_column = request_paras[RequestKey.reference_column.name]
        # drop items that have the same description
        df = df.drop_duplicates(text_column).copy()
        # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 word
        df = df[df[text_column].apply(lambda x : 0 < len(str(x).split()) < 40)]
        image_paths = []   
        image_titles = []
        for index, row in df.iterrows():
            image_name = str(row[image_column]) + ".jpg"
            image_path = os.path.join(image_folder, image_name)
            image_title = str(row[text_column])
            if os.path.exists(image_path):
                image_paths.append(image_path)
                image_titles.append(image_title)
        return image_paths, image_titles
    
    def _fine_tune(self, image_paths, image_titles, company_code):
        import torch
        import torch.nn as nn
        from transformers import CLIPProcessor, CLIPModel
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        model_name ="patrickjohncyh/fashion-clip"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        # Choose computation device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        dataset = image_title_dataset(image_paths, image_titles)
        train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=0.2)
        # Specify the loss function for images
        loss_img = nn.CrossEntropyLoss()
        # Specify the loss function for text
        loss_txt = nn.CrossEntropyLoss()
        model.to(device)
        num_epochs = self._num_epochs
        for epoch in range(num_epochs):  # Number of epochs
            pbar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch_image_paths, texts in pbar:
                images = [Image.open(image_path).convert("RGB") for image_path in batch_image_paths]
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                # Create labels
                labels = torch.arange(len(images)).to(logits_per_image.device)
                # Compute loss
                total_loss = (loss_img(logits_per_image, labels) + loss_txt(logits_per_text, labels)) / 2
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # Update the progress bar with the current epoch and loss
                pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
        # Save the fine-tuned model
        model_saved_path = f"{self._root_directory}/{company_code}/hf_clip_fine_turn"
        model.save_pretrained(model_saved_path)
        processor.save_pretrained(model_saved_path)

    def _get_vectorizer(self, company_code):
        model_saved_path = f"{self._root_directory}/{company_code}/hf_clip_fine_turn"
        if (os.path.isdir(model_saved_path)):
            print("PLM CLIP")
            return PLMCLIP(model_saved_path)
            #return self._vectorizer
        else:
            return self._vectorizer

    def do(self, request_paras, company_code):
        action_type = request_paras[RequestKey.action.name]
        if len(company_code) > 0:
            collection_name = f"{company_code}_image"
        else:
            collection_name = "image"
        if (action_type == ActionType.train.name):
            self._train(request_paras,collection_name,company_code)
            return_data= "train is done."
        elif action_type == ActionType.fine_tuning.name:
            image_paths, image_titles = self._build_data(request_paras, company_code)
            self._fine_tune(image_paths, image_titles, company_code)
            return_data= "train is done."
        else:
            if RequestKey.image_64string.name in request_paras:
                image_64string = request_paras[RequestKey.image_64string.name]
                image_bytes = base64.b64decode(image_64string)
                image_path = BytesIO(image_bytes)
                return_data = self._find_by_image(image_path,collection_name,company_code)
            elif RequestKey.predict_file.name in request_paras:
                image_file = request_paras[RequestKey.predict_file.name]
                if os.path.exists(image_file):
                    image_path = image_file
                else:
                    image_path = f"{self._root_directory}/{company_code}/{image_file}"
                return_data = self._find_by_image(image_path,collection_name,company_code)
            else:
                query_text = request_paras[RequestKey.sentence.name]
                return_data = self._find_by_text(query_text,collection_name,company_code)
        return return_data
            
        
