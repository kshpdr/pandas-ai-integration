import pandas as pd
import numpy as np
import openai
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from prompts.error_correction_prompt import ErrorCorrectionPrompt
from config import Config


class AIDataFrame(pd.DataFrame):
    _metadata = ['config', 'description', 'name', 'is_df_loaded', 'cache', 'llm_agent', 'openai_model']
    
    def __init__(self, *args, config=None, description=None, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.is_df_loaded = len(self) > 0
        self.description = description
        self.config = config or Config()  # Assuming Config is some predefined default config
        self.name = name
        self.cache = {}

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
        self.openai_model = "text-davinci-003"
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

    def chat(self, prompt):
        ans = self.llm_agent.__call__(prompt)
        print(ans['intermediate_steps'][0][0].log)
        response, command = ans['output'], ans['intermediate_steps'][0][0].tool_input
        return response, command

        
