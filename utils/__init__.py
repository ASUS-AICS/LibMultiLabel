import os
from dotenv import load_dotenv

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')


# loads environment variables from `.env` for the service.
load_dotenv(os.path.join(ROOT_DIR, '.env'))