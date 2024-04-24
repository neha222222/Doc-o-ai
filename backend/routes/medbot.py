from fastapi import APIRouter, Query
import pathlib
import textwrap
import os
import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = "AIzaSyDiOsfA1lV7Wj6DE6uX9PpOzO0-OUwVDwA"

genai.configure(api_key=GOOGLE_API_KEY)
def gemini_response(query):

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    print("i was called , gemini respinse is ready")
    return response.text



medbot_api_router = APIRouter()
@medbot_api_router.get("/bot")
def medbot(text:str):
    prefix="""answer considering yourself as a doctor and if it is not relevant to doctor background say that it is not a relevant question for a doctor. Here is a "query": " """
    suffix="""  " Summarise your answer"""
    print("getting gemini response")
    return {'medboat':gemini_response(prefix+text+suffix)}


if __name__ == "__main__":
    print(gemini_response("hello_gemini"))