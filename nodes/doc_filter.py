from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from typing import TypedDict, Annotated, Optional

from helpers.constants import FILTLLM
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
Task: is the <background_document> `relevant` to the query? Being `relevant` means that the <background_document> will help at least in part to answer <query>. It does not have to answer the entire <query> to be `relevant`.
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

class IsRelevantWithExtractEntities(TypedDict):
    b: Annotated[bool, ..., 'Is the <background_information> relevant to the <query>?']
    e: Annotated[Optional[list[str]], ..., 'if relevant, extract the key entities from the <background_information> that MUST be included in the final answer to the <query>']

class IsRelevant(TypedDict):
    b: Annotated[bool, ..., 'Is the <background_information> relevant to the <query>?']

@chain
async def __doc_filter_chain(_input: dict):
    query, document, extract_entities, filter_on_metadata = _input['rephrased_query'], _input['document'], _input['extract_entities'], _input['filter_on_metadata']
    if extract_entities:
        _chain = filter_prompt | FILTLLM.with_structured_output(schema=IsRelevantWithExtractEntities, method='json_schema', strict=True)
    else:
        _chain = filter_prompt | FILTLLM.with_structured_output(schema=IsRelevant, method='json_schema', strict=True)
    # split docs and integrate metadata
    split_docs = markdown_splitter_with_header.split_text(document.page_content)
    for s in split_docs:
        s.metadata = {**document.metadata, **{k: v for k, v in s.metadata.items() if k not in document.metadata}}

    # filter based on metadata only. keep YES, continue to filter NO based on full text
    metadata_filter = []
    if filter_on_metadata:
        metadata_filter = await _chain.abatch([
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
    doc_filter = []
    if rejected_splits_on_metadata:
        doc_filter = await _chain.abatch([
            {
                'question': query,
                'doc_contents': f'Document metadata:\n{s.metadata}\n\nDocument:\n{s.page_content}'
            }
            for s in rejected_splits_on_metadata])
        filtered_on_doc = [s for s, f in zip(rejected_splits_on_metadata, doc_filter) if f['b']]
    
    # reconstruct docs
    filtered_docs = filtered_on_metadata + filtered_on_doc
    if not filtered_docs:
        filtered = None
    else:
        filtered = Document(
            page_content='\n\n'.join([s.page_content for s in filtered_docs]),
            metadata=document.metadata)

    # get extracted key entities
    if extract_entities:
        key_entities = [e.upper() for f in metadata_filter+doc_filter if f['b'] for e in f['e']]
    else:
        key_entities = []

    return {'filtered': filtered, 'key_entities': key_entities}

async def doc_filter(filter_query: str, document: Document, extract_entities: bool, filter_on_metadata: bool):
    return await __doc_filter_chain.ainvoke({
        'rephrased_query': filter_query,
        'document': document,
        'extract_entities': extract_entities,
        'filter_on_metadata': filter_on_metadata
    })

def batch_doc_filter(filter_query: str, documents: list[Document], extract_entities: bool, filter_on_metadata: bool):
    inputs = [{'rephrased_query': filter_query, 'document': doc, 'extract_entities': extract_entities, 'filter_on_metadata': filter_on_metadata} for doc in documents]
    ret = __doc_filter_chain.batch(inputs, config={"max_concurrency": 500})  # .map().invoke(inputs)
    docs = [r['filtered'] for r in ret if r['filtered']]
    if extract_entities:
        key_entities = [e for r in ret for e in r['key_entities'] if r['key_entities']]
    else:
        key_entities = []
    return {"key_entities": key_entities, "docs": docs}