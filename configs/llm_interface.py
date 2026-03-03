from google import genai

apikey = ""
client = genai.Client(api_key=apikey)

def query_gemini(prompt, model_name="gemini-3-flash-preview"):
    response = client.models.generate_content(
        model = model_name,
        contents = [prompt]
    )
    
    return response.text