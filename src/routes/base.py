from fastapi import APIRouter, Depends
from helpers.config import get_settings, settings

base_router = APIRouter()

@base_router.get("/")
async def welcome(app_settings : settings = Depends(get_settings)):

    project_name = app_settings.APP_NAME
    version = app_settings.APP_VERSION

    return {"message": f"Welcome to {project_name} {version}"}

