from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, Optional

from helpers.constants import OUTLINELLM

system_template = \
"""Here are relevant background information to help you:

{sources}

Based on the user's query, generate a list of key topics that should be covered in the answer."""


outline_template = \
"""Now, your next task is to answer my query above. But before that, organize your thoughts using your scratchpad.

Generate an outline of your response, including only category and section headings.
Do not waste space by spelling out each point in detail; you will do that in the response section later."""


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
You don't need to write out extensive rationales here; the response will be done later."""


class ScratchpadResponse(TypedDict):
    rough_outline: Annotated[list[str], ..., "a rough outline of the response - list of headings and subheadings, keep it very brief"]
    patient: Annotated[Optional[str], ..., "if applicable, POINT FORM pertinent patient information"]
    factors: Annotated[Optional[str], ..., "if applicable, POINT FORM the most important factors to consider when making a recommendation"]
    conservative: Annotated[Optional[str], ..., "if applicable, POINT FORM conservative management options"]
    medical: Annotated[Optional[str], ..., "if applicable, POINT FORM medical management options"]
    surgical: Annotated[Optional[str], ..., "if applicable, POINT FORM surgical management options"]



prompt = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('human', '{query}'),
    ('ai', '{key_topics}'),
    ('human', outline_template)
])

outline_chain = prompt | OUTLINELLM.with_structured_output(ScratchpadResponse, method='json_schema', strict=True)