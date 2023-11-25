from openai import OpenAI
from config import Config

class QueryHistory:
    def __init__(self):
        self._history = []

    def add_query(self, query, response, command):
        self._history.append({'query': query, 'response': response, 'command': command})

    def get_last_query(self):
        return str(self._history[-1]) if self._history else ""

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history
    
    def is_query_semantically_similar(self, new_query):
        config = Config()
        history_str = "\n".join([f"Index: {index}\nQuery: {entry['query']}\nResponse: {entry['response']}\nCommand: {entry['command']}" 
                                for index, entry in enumerate(self._history)])
        prompt = f"You will be provided with different queries, corresponding responsed and commands. You have to determine whether the following query is semantically similar to any of the above queries? Say just 'No', if there are no similar. Say 'Yes, X' if there is one. X should be index of the query (remember, in python indices start from 0). \n\nQuery: {new_query}\n"

        client = OpenAI(
            api_key=config.get_open_ai_key(),
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": prompt
                },
                {
                    "role": "user",
                    "content": history_str 
                }
            ],
            max_tokens=256,
            n=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print(history_str)
        # print("openai response: ", response)
        result = response.choices[0].message.content

        if result.lower().startswith("yes"):
            query = self._history[int(result.split(",")[1].strip())]
            return query
        else:
            return False
            
global_query_history = QueryHistory()