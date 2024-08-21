from helpers.constants import EDUCATION_LEVEL
from helpers.template import surg_abbrevs_table, main_conv_sys_template, memory_template

from langchain_core.prompts import ChatPromptTemplate


guiding_prompt = \
f"""Since different guidelines use different terms to refer to the same treatment, always unify them using this reference table:

{surg_abbrevs_table}

You should try to combine the information from each guideline where possible, highlighting only major differences as you go, if applicable.
You should give a brief comparison between the guidelines (CUA, AUA, EAU), at the end of your response.

If discussing management options, consider whether any conservative/lifestyle, medical/pharmacological, and procedural/surgical options are applicable.
You do not have to discuss all options if not applicable, but you should at least have this mindset when considering the options."""

human_hidden_instructions = \
f"""Good outline, adhere to it and answer the query.

Hidden Instructions: If there are conflicting recommendations, place your HIGHEST PRIORITY ON PATIENT SAFETY (e.g. risk of significant bleeding).
YOU MUST DO THIS AND NOT FORGET THIS. Always take a moment to think before you respond. You must NOT make unsafe recommendations.

Education Level Hint: {EDUCATION_LEVEL}

You should include all reasonable discussion topics in your response. At the end, include a brief comparison between the guidelines (CUA, AUA, EAU) if applicable. Include a brief recommendation at the end with a rationale if applicable.
                          
One more thing: for any CLAIM you make, you MUST provide a reference in this format at the end of the claim: <doc_#x>, <doc_#y>, ..."""


response_format = \
"""provide your answer in this json schema format:
{{
    "scratchpad": "string",
    "final_answer": "string"
}}

<scratchpad_instructions>
Use this area to organize your thoughts before you answer the question.

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
opt: ALL possible options available to the patient - be inclusive
</scratchpad_instructions>"""


def synthesize_prompt(context, outline, memory, query, rephrased):
    prompt = ChatPromptTemplate.from_messages([
        ('system', main_conv_sys_template + guiding_prompt),
        #('ai', memory_template),
        #('system', guiding_prompt),
        ('human', query),
        #('ai', f'Based on the query and context, I think the user is asking: {rephrased}'),
        #('ai', f'Based on the query and context, here is my response outline: \n\n{outline}'),
        ('human', human_hidden_instructions),
    ])
    return prompt.partial(context=context, summary=memory)