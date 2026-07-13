import time
import os
import openai
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()
local_api_key= os.getenv("LOCAL_API_KEY")
openai.api_key = local_api_key
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

LOCAL_MODEL = ['local_model']
BASE_URL = os.getenv("LOCAL_BASE_URL")
openai.api_base = BASE_URL

def get_token_limit(model):
    """Returns the token limitation of provided model"""
    return 16384

class LocalLLM(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
    
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150 

    def tokenize(self, text):
        """
        Approximate token count.

        using tiktoken provides a rough estimate for length checking.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")

        return num_tokens + self.redundant_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        '''
        Local LLM model does not need to prepare for inference
        '''
        pass
    
    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]

        cur_retry = 0
        num_retry = self.retry

        # Check if the input is too long
        input_length = self.tokenize(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]

        while cur_retry <= num_retry:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages= query,
                    request_timeout = 3000,
                    # tool_choice="none",
                    # tools=[],
                    # functions=[],
                    )
                result = response["choices"][0]["message"]["content"].strip() # type: ignore

                time.sleep(5)

                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("response:\n\n", response, "\n\n")
                print("Number of token: ", self.tokenize(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
