from .BaseController import BaseController
from fastapi import UploadFile, File
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import asyncio
import os, re

import logging
logger = logging.getLogger(__name__)

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.project_path = self.get_project_path()

    def validfile(self, file: UploadFile):
        if file.content_type not in self.app_settings.ALLOWED_FILE_TYPES:
            return False, 'Invalid File Type' 

        if file.size > self.app_settings.MAX_FILE_SIZE:
            return False, 'File Size Too Large'

        return True, "Valid File"
    

    def _clean_file_name(self, file_name: str):
        # Remove special chars
        clean_filename = re.sub(r"[^\w.]", "", file_name)
        return clean_filename
    
    def get_file_path(self, filename: str):
        clean_filename = self._clean_file_name(filename)
        random_key = self.random_key()

        file_path = os.path.join(self.project_path, f"{random_key}_{clean_filename}")
        if os.path.exists(file_path):
            return self.get_file_path(filename=filename)
        
        return file_path, f"{random_key}_{clean_filename}"

    async def get_file_content(self, file_id: str):
        file_ext = os.path.splitext(file_id)[1]
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_id}")
            return None

        try:
            if file_ext == '.txt':
                loader = TextLoader(file_path, encoding="utf-8")
                return await asyncio.to_thread(loader.load)

            elif file_ext == '.pdf':
                loader = PyMuPDFLoader(file_path)
                return await asyncio.to_thread(loader.load)

            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return None

        except Exception as e:
            logger.error(f"Error loading file {file_id}: {e}")
            return None