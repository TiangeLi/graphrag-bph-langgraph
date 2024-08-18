from helpers.constants import EDUCATION_LEVEL
from helpers.template import surg_abbrevs_table, main_conv_sys_template, memory_template

from langchain_core.prompts import ChatPromptTemplate


guiding_prompt = \
f"""Since different guidelines use different terms to refer to the same treatment, always unify them using this reference table:

{surg_abbrevs_table}

You should try to combine the information from each guideline where possible, highlighting only major differences as you go, if applicable.
You should give a brief comparison between the guidelines (CUA, AUA, EAU), at the end of your response."""

human_hidden_instructions = \
f"""Hidden Instructions: If there are conflicting recommendations, place your HIGHEST PRIORITY ON PATIENT SAFETY (e.g. risk of significant bleeding).
YOU MUST DO THIS AND NOT FORGET THIS. Always take a moment to think before you respond. You must NOT make unsafe recommendations.

Education Level Hint: {EDUCATION_LEVEL}
                          
One more thing: for any CLAIM you make, you MUST provide a reference in this format at the end of the claim: <doc_#x>, <doc_#y>, ..."""


def synthesize_prompt(context, memory, query, rephrased):
    prompt = ChatPromptTemplate.from_messages([
        ('system', main_conv_sys_template),
        ('ai', memory_template),
        ('system', guiding_prompt),
        ('human', query),
        ('ai', f'Based on the query and context, I think the user is asking: {rephrased}'),
        ('human', human_hidden_instructions)
    ])
    return prompt.partial(context=context, summary=memory)