import langchain
import langchain_community
import openai
from langchain.utils import get_from_dict_or_env
from pydantic import root_validator
import logging

logger = logging.getLogger(__name__)


class ChatClientAdapter(openai.ChatCompletion):

    @classmethod
    def create(cls, prompt, *args, **kwargs):
        # Pre-process
        assert len(prompt) == 1
        messages=[
            {'role': 'user', 'content': prompt[0]},
        ]

        # Core
        completion = super().create(messages=messages, *args, **kwargs)

        # Post-process
        for choice in completion['choices']:
            assert choice['message']['role'] == 'assistant'
            choice['text'] = choice['message']['content']

        logger.info(f"Response text: {response_text}")

        return completion
    

class MyOpenAI(langchain_community.llms.OpenAI):

    def __new__(cls, *args, **kwargs):
        return super(langchain_community.llms.openai.BaseOpenAI, cls).__new__(cls)

    @root_validator(pre=True)
    def validate_environment(cls, values):
        logger.info(f"Validating environment: values = {values}")
        openai_api_key = get_from_dict_or_env(values, "openai_api_key", "OPENAI_API_KEY")
        openai.api_key = openai_api_key
        values["client"] = ChatClientAdapter
        return values
    

if __name__ == '__main__':
    OPENAI_API_KEY = ''

    openai.api_key = OPENAI_API_KEY

    llm = MyOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    resp = llm("Tell me a joke")
    print(resp)
