import base64
import requests
import os
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError as e:
        logger.error(f"File not found: {image_path}")
        raise e

def prepare_payload(encoded_images: List[str]) -> Dict:
    """Prepares the payload for the API request."""
    return {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '''You are an AI assistant designed to estimate the height of a person based on three images provided. Your task is to analyze the images using your vision capabilities and any available contextual clues to make an educated guess about the person's height.

                        Instructions:

                        Image Analysis:

                        Carefully examine the three images of the person standing in different poses.
                        Use camera angles, perspective, and any visible common objects (such as phones, furniture, etc.) to help estimate the person's height.
                        Contextual Clues:

                        If the images contain common objects with known dimensions, use these objects to extrapolate and calculate the person's height.
                        Consider the relative sizes and positions of objects in the images to improve the accuracy of your estimation.
                        Combining Data:

                        Use all three images to cross-reference and refine your height estimation.
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
        raise e

def main(image_paths: List[str]):
    """Main function to encode images, prepare payload, and send request to the API."""
    encoded_images = [encode_image(image_path) for image_path in image_paths]
    payload = prepare_payload(encoded_images)
    response = send_request_to_api(payload)
    final_res = response['choices'][0]['message']['content']
    logger.info(f"Assistant Response: {final_res}")
    print(final_res)

if __name__ == "__main__":
    # Paths to your images
    image_paths = ["IMG_1455.JPG", "IMG_6540.jpg", "IMG_7370.jpg"]
    main(image_paths)
