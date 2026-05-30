import time
import os
from google import genai
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def get_token_limit(model="gemini-3.1-flash-lite"):
    """Returns approximate token limits for Gemini models."""

    limits = {
        "gemini-3.1-flash-lite": 1048576,
        "gemini-2.5-pro": 1048576,
        "gemini-2.5-flash": 1048576,
        "gemini-1.5-pro": 2097152,
        "gemini-1.5-flash": 1048576,
    }

    if model not in limits:
        raise NotImplementedError(
            f"get_token_limit() is not implemented for model {model}."
        )

    return limits[model]


class Gemini(BaseLanguageModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--retry",
            type=int,
            help="retry time",
            default=5,
        )

    def __init__(self, args):
        super().__init__(args)

        self.retry = args.retry
        self.model_name = args.model_name

        self.maximum_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150

    def tokenize(self, text):
        """
        Approximate token count.

        Gemini has its own tokenizer, but using tiktoken
        provides a rough estimate for length checking.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")

        return num_tokens + self.redundant_tokens

    def prepare_for_inference(self, model_kwargs={}):
        """
        Gemini model does not need preparation.
        """
        pass

    def generate_sentence(self, llm_input):

        cur_retry = 0
        num_retry = self.retry

        input_length = self.tokenize(llm_input)

        if input_length > self.maximum_token:
            print(
                f"Input length {input_length} is too long. "
                f"Maximum token limit is {self.maximum_token}.\n"
                f"Right truncate the input."
            )

            # crude truncation (same behavior as original)
            llm_input = llm_input[: self.maximum_token]

        while cur_retry <= num_retry:

            try:

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=llm_input,
                )

                return response.text.strip()

            except Exception as e:

                print("Message:", llm_input)
                print("Number of token:", self.tokenize(llm_input))
                print(e)

                time.sleep(30)
                cur_retry += 1

        return None