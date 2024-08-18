from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from helpers.llm_response_types import BooleanResponse
from helpers.constants import FILTLLM, ADD_METADATA_FILTER
from helpers.template import surg_abbrevs_table

headers_to_split_on = [
    ("#", "Title"),
    ("##", "Header 1"),
    ("###", "Header 2"),
    ("####", "Header 3"),
    ("#####", "Header 4"),
    ("######", "Header 5"),
    ("#######", "Header 6")
]

markdown_splitter_with_header = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False)

sys_template = \
f"""Use this abbreviations table to help you understand the background information:
{surg_abbrevs_table}

Evaluate the <background_document> against the <query>.
Task: is the <background_document> `relevant` to the query?
    - You must consider each ENTITY in the Query.
    - You must consider each FOCUS/TOPIC of the <background_document>.
    - The <background_document> is `relevant` if the <background_document>'s primary FOCUS contains information pertinent to ANY ONE of the entities in the <query> (including any Examples, Equivalents, Brand Names, Abbreviations, etc.)

Remember: ONLY use the <background_document> to inform your decision."""

hum_template = \
"""<background_document>
{doc_contents}
</background_document>

<query>
{question}
</query>"""


filter_prompt = ChatPromptTemplate.from_messages(
    [('system', sys_template),
    ('human', hum_template)])

@chain
def __doc_filter_chain(_input: dict):
    query, document, add_metadata_filter = _input['rephrased_query'], _input['document'], _input['add_metadata_filter']
    _chain = filter_prompt | FILTLLM.with_structured_output(schema=BooleanResponse, method='json_schema', strict=True)

    # split docs and integrate metadata
    split_docs = markdown_splitter_with_header.split_text(document.page_content)
    for s in split_docs:
        s.metadata = {**document.metadata, **{k: v for k, v in s.metadata.items() if k not in document.metadata}}

    # filter based on metadata only. keep YES, continue to filter NO based on full text
    if add_metadata_filter:
        metadata_filter = _chain.batch([
            {
                'question': query,
                'doc_contents': s.metadata
            }
            for s in split_docs])
        rejected_splits_on_metadata = [s for s, f in zip(split_docs, metadata_filter) if not f['b']]
        filtered_on_metadata = [s for s, f in zip(split_docs, metadata_filter) if f['b']]
    else:
        # if not filtering on metadata, we would still want to do full text filtering even if metadata filtering is disabled
        rejected_splits_on_metadata = split_docs
        filtered_on_metadata = []

    # filter rejected docs, now based on full text
    filtered_on_doc = []
    if rejected_splits_on_metadata:
        doc_filter = _chain.batch([
            {
                'question': query,
                'doc_contents': f'Document metadata:\n{s.metadata}\n\nDocument:\n{s.page_content}'
            }
            for s in rejected_splits_on_metadata])
        filtered_on_doc = [s for s, f in zip(rejected_splits_on_metadata, doc_filter) if f['b']]
    
    # reconstruct docs
    filtered_docs = filtered_on_metadata + filtered_on_doc
    if not filtered_docs:
        return None
    filtered = Document(
        page_content='\n\n'.join([s.page_content for s in filtered_docs]),
        metadata=document.metadata
    )
    return filtered

def batch_doc_filter(rephrased_query: str, documents: list[Document]):
    inputs = [{'rephrased_query': rephrased_query, 'document': doc, 'add_metadata_filter': ADD_METADATA_FILTER} for doc in documents]
    filtered = __doc_filter_chain.map().invoke(inputs) # vs .batch(inputs)
    return [doc for doc in filtered if doc]