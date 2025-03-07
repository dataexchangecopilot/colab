"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.

arguments example: 
    '{\"action\": \"train\",\"training_file\": \"plm_menu_test.csv\",\"column_name\": \"menu\"}'
    '{\"action\": \"search\",\"sentence\": \"What is Quick Search\"}'

when training
send a file, tell which column is for searching

when searching
    query is from sentence 
    search result orginal + score are returned
"""
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os, base64
from open_webui.backend_customer.plm_enum import RequestKey, ActionType, ResponseKey, UseCase
from open_webui.backend_customer.config import *

class PLMSearchText:
    """ search text /sentence """
    def __init__(self) -> None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._embeddings = HuggingFaceEmbeddings(model_name = model_name)
        self._score_column = "score"
        self._persist_directory = DB_DIRECTORY 
        self._root_directory = ROOT_DIRECTORY

    # delete collection
    def _delete_collection(self, collection_name):
        db = Chroma(collection_name= collection_name, 
            persist_directory= self._persist_directory,
            embedding_function= self._embeddings
        )
        db.similarity_search(query="any")
        db.delete_collection()

    def _train(self, input_file, sentence_column, collection_name):
        _, file_extension = os.path.splitext(input_file)
        if file_extension.startswith(".xls"):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)
        # copy sentence column, so saved to metadata
        # then use score column as page_content
        df[self._score_column] = df[sentence_column]

        loader = DataFrameLoader(data_frame= df, 
            page_content_column= self._score_column
        )
        docs = loader.load()
        # delete collection if exist
        self._delete_collection(collection_name)

        Chroma.from_documents(docs, 
            embedding= self._embeddings, 
            collection_name= collection_name, 
            persist_directory= self._persist_directory
        )
        #print(docs[0])
    
    # Try to get the item which has an integer as key
    def _try_get_key(self, string_with_space):
        string_list = string_with_space.split(' ')
        key_list =[]
        for item in string_list:
            if any(char.isdigit() for char in item):
                # If successful, add it to the list
                key_list.append(item)
        return key_list 
    
    def _build_where(self, key_list):
        contain_list = []
        for key in key_list:
            contain = {"$contains": key +","}
            contain_list.append(contain)
        if len(contain_list) == 1:
            return contain_list[0]
        else:
            return {"$and": contain_list}
    # do sentence similarity
    def _search(self, query, collection_name):
        db = Chroma(collection_name = collection_name, 
            persist_directory= self._persist_directory,
            embedding_function= self._embeddings
        )
        
        key_list = self._try_get_key(query)

        if len(key_list) > 0:
            docs_with_score = db.similarity_search_with_relevance_scores(query= query, k=15,where_document = self._build_where(key_list))
        else:
            docs_with_score = db.similarity_search_with_relevance_scores(query= query, k=15)    
        
        results =[]
        for doc, score in docs_with_score:
            result = doc.metadata
            result[self._score_column] = "{:.4f}".format(score)
            #print(result)
            results.append(result)
        return results

    def do(self, request_paras, company_code):
        action_type = request_paras[RequestKey.action.name]
        collection_name = request_paras[RequestKey.category.name]
        if len(company_code) > 0:
            company_collection_name = f"{company_code}_{collection_name}"
        else:
            company_collection_name = collection_name
        if (action_type == ActionType.train.name):
            file_name = request_paras[RequestKey.file_name.name]
            content_64string = request_paras[RequestKey.content_64string.name]
            column_name = request_paras[RequestKey.column_name.name]
            # format train_file path
            train_file = f"{self._root_directory}/{company_code}/{file_name}"
            content_bytes = base64.b64decode(content_64string)
            with open(train_file, "wb") as fw:
                fw.write(content_bytes)
            self._train(train_file, column_name, company_collection_name)
            return_data= "train is done."
        else:
            sentence = request_paras[RequestKey.sentence.name]
            results = self._search(sentence,company_collection_name)
            return_data = {
                ResponseKey.sentence.name:sentence, 
                ResponseKey.category.name:collection_name, 
                ResponseKey.results.name:results
            }  
        return return_data
    
    def get_category_datas(self, category_name):
        db = Chroma(collection_name = category_name, 
            persist_directory= self._persist_directory,
            embedding_function= self._embeddings
        )
        dict_category = db.get()
        return dict_category["metadatas"]

