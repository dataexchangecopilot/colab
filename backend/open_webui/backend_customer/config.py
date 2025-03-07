import os
from dotenv import load_dotenv

# tensorflow/core/util/port.cc:113] oneDNN custom operations are on. 
# You may see slightly different numerical results due to floating-point round-off errors from different computation orders. 
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# https://stackoverflow.com/questions/78530756/error-only-instances-of-keras-layer-can-be-added-to-a-sequential-model
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
# load config from .env file
load_dotenv()

# vector index persist directory
DB_DIRECTORY = os.getenv('DB_DIRECTORY', "./data/chromadb")

# root Directory to scrape
ROOT_DIRECTORY =  os.getenv('ROOT_DIRECTORY', "./data/root_repos")

#plm search image
# IMAGE_WIDTH = os.getenv("IMAGE_WIDTH",255)
# IMAGE_HEIGHT = os.getenv("IMAGE_HEIGHT",255)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY',"")

GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS',"")