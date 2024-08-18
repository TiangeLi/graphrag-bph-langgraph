from langchain_core.prompts import ChatPromptTemplate

from helpers.llm_response_types import BooleanResponse
from helpers.constants import ROUTERLLM
from helpers.template import meds_abbrevs_table, surg_abbrevs_table, other_abbrevs_table


router_template = \
f"""<related_terms>
{meds_abbrevs_table}

{surg_abbrevs_table}

{other_abbrevs_table}
</related_terms>

<conversation_summary>
{{summary}}
</conversation_summary>

Evaluate the user query. Is it about benign prostate hyperplasia (BPH) and therefore requires information from a BPH knowledge base?

You would want to use the BPH knowledge base, if:
the user query is or could be about benign prostate hyperplasia (BPH), including definitions, symptoms, diagnosis, testing, management, treatment, related medications/medical therapy, surgical therapy, complications, side effects, or other topics broadly related to BPH."""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', router_template),
        ('human', 'Query: {question}')
    ]

)

router_chain = router_prompt | ROUTERLLM.with_structured_output(schema=BooleanResponse, method='json_schema', strict=True)