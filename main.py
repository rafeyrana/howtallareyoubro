import base64
import os
import logging
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image(image_file: UploadFile) -> str:
    """Encodes an image to base64."""
    try:
        return base64.b64encode(image_file.file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise HTTPException(status_code=500, detail="Error encoding image")

def prepare_payload(encoded_images: List[str]) -> Dict:
    """Prepares the payload for the API request."""
    return {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '''You are an AI assistant designed to estimate the height of a person based on four images provided. Your task is to analyze the images using your vision capabilities and any available contextual clues to make an educated guess about the person's height.

                        Instructions:

                        Image Analysis:

                        Carefully examine the four images of the person standing in different poses.
                        Use camera angles, perspective, and any visible common objects (such as phones, furniture, etc.) to help estimate the person's height.
                        Contextual Clues:

                        If the images contain common objects with known dimensions, use these objects to extrapolate and calculate the person's height.
                        Consider the relative sizes and positions of objects in the images to improve the accuracy of your estimation.
                        Combining Data:

                        Use all four images to cross-reference and refine your height estimation.
                        Look for consistency in the person's height across different poses and angles.
                        Output:

                        Always provide the estimated height in inches.

                        Your output should only be an integer which is the estimated height."''',
                    }
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    }
                    for encoded_image in encoded_images
                ],
            }
        ],
        "max_tokens": 300,
    }

def send_request_to_api(payload: Dict) -> Dict:
    """Sends the payload to the OpenAI API and returns the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail="API request failed")

@app.post("/estimate-height/")
async def estimate_height(images: List[UploadFile] = File(...)):
    """Endpoint to estimate the height of a person based on four images."""
    if len(images) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 images are required")

    encoded_images = [encode_image(image) for image in images]
    print(f"encoded_images: {len(encoded_images)}")
    payload = prepare_payload(encoded_images)
    response = send_request_to_api(payload)
    print(f"model response: {response}")
    final_res = response['choices'][0]['message']['content']
    logger.info(f"Assistant Response: {final_res}")
    match = re.search(r'\b(\d+)\b\s*', final_res)
    if match:
        return  {"estimated_height": int(match.group(1))}
    else:
        return {"estimated_height": final_res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
