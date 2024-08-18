from langchain_core.prompts import ChatPromptTemplate

from typing import TypedDict, Annotated

from helpers.constants import CONVLLM

scratchpad_instructions = \
"""Use this area to organize your thoughts before you answer the question.

The scratchpad has LIMITED SPACE, so you must be efficient!

RULES:
I recommend that you use shortforms and abbreviations where possible to keep your thoughts as tight as possible.
BUT: do NOT sacrifice brevity for inclusiveness. You must include all relevant information, even if it means using more space on the scratchpad.

Example:
Instead of writing "National Institute of Health", you could write "NIH".
Instead of writing "The patient has a history of hypertension", you could write "HTN".
Instead of "anticoagulation", you could write "ac".

Remember: the scratchpad is for YOUR use only. It will not be visible to the user, so you can use it to structure your thoughts in a way that makes sense to you; it's only purpose is to help you think before you answer.
You don't need to write out extensive rationales here; that's for the response section.

Suggested sections:
pt: pertinent information about the patient
fctr: the most important <FACTOR>s to consider when making a recommendation
opt: ALL possible options available to the patient - be inclusive"""


class Response(TypedDict):
    scratchpad: Annotated[str, ..., scratchpad_instructions]
    final_answer: str

chat_llm_w_scratchpad = CONVLLM.with_structured_output(Response, method='json_schema', strict=True)