import sys
import os
import time

current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from datetime import datetime
from dataclasses import dataclass, field

from utils.rag_helper import RAGBase

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    #total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model: str, usage: dict) -> float:
    """
    Calculate the cost of an LLM call based on the model and token usage.
    """
    cost = 0
    if "claude-haiku-4-5" in model:
        cost = (usage.input_tokens * 1.00 + usage.output_tokens * 5.00) / 1000000
    return cost


class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.content[0].text
    
    def _call_llm(self, prompt):
        input_messages = [
            #{"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=input_messages
        )
        return response
    
    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.content[0].text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            #total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )
    
        print(call_record)
        self.last_call = call_record
