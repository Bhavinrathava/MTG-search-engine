import json
import os
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GPTInterface:
    """Manages interactions with OpenAI's GPT models"""
    
    def __init__(self):
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        
        # Default parameters
        self.default_model = "gpt-4o-mini"
        self.default_max_tokens = 500
        self.default_temperature = 0.7
    
    def call_gpt(self, 
                 prompt: str,
                 model: str = None,
                 max_tokens: int = None,
                 temperature: float = None) -> Dict[str, Any]:
        """
        Call GPT model with the given prompt and parameters
        
        Args:
            prompt: The input text to send to GPT
            model: The GPT model to use (defaults to self.default_model)
            max_tokens: Maximum number of tokens in the response
            temperature: Controls randomness in the response
            
        Returns:
            Dict containing the parsed JSON response from GPT
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return {"error": str(e)}
    
    def generate_query_parameters(self, 
                                original_query: str,
                                expanded_query: str,
                                card_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate MongoDB query parameters using GPT
        
        Args:
            original_query: The original user query
            expanded_query: The expanded version of the query
            card_template: Template showing card object structure
            
        Returns:
            Dict containing MongoDB query parameters
        """
        prompt = f'''
        You are a Magic The Gathering Card Expert System. You will be given a user query where the user is looking for a specific type of card or a specific card. 
        You will be given the following : 
        - Card Object Template explaining what each field means for a card object stored in DB. 
        - Original User Query in String
        - Expanded User Query in String

        The Cards are stored in a MongoDB Collection. The intent is to use collection.find(query) based on user preferences.
        It is okay if the user query is vague, you have to try to convert user query into a quantifiable query for MongoDB. Is is okay if the MongoDB Query is not Specific. 

        Your task is to return a JSON representing the query object that is to be passed to the MongoDB's collection.find(query) command.  Keep the query parameters to bare essentials. No need to specify exists and other bare-bone query parameters. Just filters are fine. 

        Here is the Card Template :

        {card_template}

        Here is the User Query : 

        {original_query}

        Here is the expanded Query : 

        {expanded_query}
        '''
        
        return self.call_gpt(prompt)
