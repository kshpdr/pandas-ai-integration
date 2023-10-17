import pandas as pd
import numpy as np
import openai
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from prompts.error_correction_prompt import ErrorCorrectionPrompt
from config import Config
from datastructure.history_manager import global_query_history

class AIDataFrame(pd.DataFrame):
    _metadata = ['config', 'description', 'name', 'is_df_loaded', 'cache', 'llm_agent', 'openai_model']
    
    def __init__(self, *args, config=None, description=None, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.is_df_loaded = len(self) > 0
        self.description = description
        self.config = config or Config()  # Assuming Config is some predefined default config
        self.name = name
        self.cache = {}

    def add_to_history(self, query, response, code):
        global_query_history.add_query(query, response, code)

    def get_last_query(self):
        return global_query_history.get_last_query()
    
    def wrap_with_last_query(self, query):
        new_query = "Previous query: " + self.get_last_query()
        new_query += f"Current query: {query}"
        return new_query

    @property
    def col_count(self):
        if self.is_df_loaded:
            return len(list(self.columns))
        
    @property
    def row_count(self):
        if self.is_df_loaded:
            return len(self)
        
    @property
    def sample_head_csv(self):
        if self.is_df_loaded:
            return self.head(5).to_csv()
        
    
    @property
    def metadata(self):
        return self.info()
    
    def to_csv(self, file_path):
        self.to_csv(file_path)
    

    def clear_cache(self):
        self.cache = {}
    
        
    def initialize_middleware(self):
        open_ai_key = self.config.get_open_ai_key()

        self.llm_agent = create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=open_ai_key), \
                                        self, verbose=False, return_intermediate_steps=True)
        openai.api_key = open_ai_key
        self.openai_model = "gpt-4"
        return
    
    def query_dataframe(self, query):
        if query not in self.cache:
            ans = self.llm_agent.run(query)
            self.cache[query] = ans
        else:
            ans= self.cache[query]
        return ans
    
    def code_error_correction(self, query, error, old_python_code):
        prompt = ErrorCorrectionPrompt().get_prompt(self.pd_df, query, error, old_python_code)
        #print(prompt)
        response = openai.Completion.create(engine = self.openai_model, prompt = prompt)
        answer = response.choices[0].text

        return answer

    def chat(self, prompts):
        ans = self.llm_agent.__call__(prompts)
        response, command = ans['output'], ans['intermediate_steps'][0][0].tool_input
        # uncomment to see log of the langchain
        # print(f"FULL LOG: {ans}")
        # print(ans['intermediate_steps'][0][0].log)

        return response, command

        
