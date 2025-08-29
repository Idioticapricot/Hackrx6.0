# src/ai/gemini_client.py
"""Gemini API client for text and vision processing"""

import httpx
import base64
from typing import List, Dict, Any, Optional
from ..core.config import config

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self):
        self.api_key = config.api.gemini_key
        self.base_url = config.api.gemini_url
        self.timeout = config.api.llm_timeout
    
    async def generate_text(self, prompt: str, system_prompt: str = "", model: str = None) -> str:
        """Generate text using Gemini API"""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        model_name = model or config.models.llm_model
        url = f"{self.base_url}/{model_name}:generateContent"
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048,
                "topP": 0.9,
                "topK": 40
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        headers = config.get_api_headers()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                print(f"🔍 Gemini API Response: {result}")
                
                if "candidates" in result and result["candidates"]:
                    candidate = result["candidates"][0]
                    
                    # Check for finish reason
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason == "SAFETY":
                        return "Content was blocked by safety filters. Please try rephrasing your request or use different language."
                    elif finish_reason == "RECITATION":
                        return "Content was blocked due to recitation concerns. Please try a different approach."
                    elif finish_reason == "MAX_TOKENS":
                        return "Response was truncated due to length limits. Please ask for a shorter summary."
                    
                    if "content" in candidate and "parts" in candidate["content"]:
                        if candidate["content"]["parts"]:
                            return candidate["content"]["parts"][0]["text"]
                    
                    # If no content but candidate exists, check for other issues
                    if "content" not in candidate:
                        return "No content generated. This might be due to safety filters or content policy restrictions."
                    
                    return "Unable to generate response - content blocked or empty"
                elif "error" in result:
                    error_msg = result['error'].get('message', 'Unknown error')
                    error_code = result['error'].get('code', 'No code')
                    return f"API Error ({error_code}): {error_msg}"
                else:
                    return "No response generated from Gemini API"
                    
            except httpx.TimeoutException:
                return f"Error: Request timed out after {self.timeout} seconds."
            except httpx.HTTPStatusError as e:
                return f"Error: HTTP {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Error: {str(e)}"
    
    async def analyze_image(self, image_url: str, prompt: str, system_prompt: str = "") -> str:
        """Analyze image using Gemini Vision API"""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        model_name = config.models.llm_vision_model
        url = f"{self.base_url}/{model_name}:generateContent"
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # For image URLs, we need to fetch and encode the image
        try:
            async with httpx.AsyncClient() as client:
                img_response = await client.get(image_url, timeout=30)
                img_response.raise_for_status()
                image_data = base64.b64encode(img_response.content).decode()
                
                # Determine MIME type from URL or content
                mime_type = "image/jpeg"
                if image_url.lower().endswith('.png'):
                    mime_type = "image/png"
                elif image_url.lower().endswith('.webp'):
                    mime_type = "image/webp"
                
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": full_prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_data
                                }
                            }
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048,
                        "topP": 0.9,
                        "topK": 40
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                }
                
                headers = config.get_api_headers()
                
                response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                print(f"🔍 Gemini Vision API Response: {result}")
                
                if "candidates" in result and result["candidates"]:
                    candidate = result["candidates"][0]
                    
                    # Check for finish reason
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason == "SAFETY":
                        return "Content was blocked by safety filters. Please try rephrasing your request."
                    elif finish_reason == "RECITATION":
                        return "Content was blocked due to recitation concerns. Please try a different approach."
                    
                    if "content" in candidate and "parts" in candidate["content"]:
                        if candidate["content"]["parts"]:
                            return candidate["content"]["parts"][0]["text"]
                    return "Unable to analyze image - content blocked or empty"
                elif "error" in result:
                    return f"API Error: {result['error'].get('message', 'Unknown error')}"
                else:
                    return "No response generated from Gemini API"
                    
        except httpx.TimeoutException:
            return f"Error: Request timed out after {self.timeout} seconds."
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

# Global Gemini client instance
gemini_client = GeminiClient()

async def generate_text_with_gemini(prompt: str, system_prompt: str = "", model: str = None) -> str:
    """Generate text using Gemini API"""
    return await gemini_client.generate_text(prompt, system_prompt, model)

async def analyze_image_with_gemini(image_url: str, prompt: str, system_prompt: str = "") -> str:
    """Analyze image using Gemini Vision API"""
    return await gemini_client.analyze_image(image_url, prompt, system_prompt)