from helpers.config import get_settings, settings
import os
import random 
import string

class BaseController:

    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.files_dir = os.path.join(self.base_dir, "assets/files")
        self.vdb_dir = os.path.join(self.base_dir, "assets/vdb")

    def random_key(self, length: int=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def get_vdb_path(self, db_name: str) -> str:
        db_path = os.path.join(self.vdb_dir, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        return db_path
    
    def get_project_path(self):
        project_path = os.path.join(self.files_dir)
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        return project_path