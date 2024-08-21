from dotenv import load_dotenv
load_dotenv('.env', override=True)

from concurrent.futures import ThreadPoolExecutor
from math import ceil as math_ceil

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_transformers import LongContextReorder, EmbeddingsClusteringFilter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from typing import TypedDict, Annotated, Optional

from helpers.constants import (
    CLARIFYLLM, FILTER_ON_METADATA,
    LARGE_EMBD,
    CLUSTER_SIZE,
    TOP_ENTITIES_PER_QUERY_ON_BLIND_PASS,
    TOP_ENTITIES_PER_QUERY_ON_SECOND_PASS,
    TOP_K_CHUNKS_PER_SIM_SEARCH_ON_BLIND_PASS,
    TOP_K_CHUNKS_PER_SIM_SEARCH_ON_SECOND_PASS,
    TOP_K_CHUNKS_FOR_ENTITY_SEARCH
)

from helpers.utils import reciprocal_rank_fusion

from nodes.router import router_chain
from nodes.db_retriever import DBRetriever
from nodes.doc_filter import doc_filter
from nodes.outliner import outline_chain
from nodes.prompt_synthesizer import synthesize_prompt
from nodes.chat import chat_llm_w_scratchpad

# --- utilities ---
db_retriever = DBRetriever()
doc_reorder = LongContextReorder()
GUIDELINE_NAMES = ['CUA', 'AUA', 'EAU']

# --- graph reducer functions ---
def graph_overwrite(_, new):
    return new

def graph_extend(old, new):
    if new == "__RESET__":
        return []
    for n in new:
        if n not in old:
            old.append(n)
    return old

# --- basic graph elements ---
class FPayload(TypedDict):
    to_filter: Document
    filter_query: str
    extract_entities: bool
class FilterState(TypedDict):
    payload: FPayload


class State(TypedDict):
    fpayloads: Annotated[list[FPayload], graph_overwrite]
    filtered: Annotated[list[Document], graph_extend]

    prompt: str
    summary: str
    use_guidelines: bool
    use_knowledge_graph: bool

    multiqueries: Annotated[list[str], graph_overwrite]

    num_times_done_sim_search: int

    chunks_to_exclude_on_second_pass: Annotated[list[str], graph_overwrite]
    confirmed_chunks: Annotated[list[Document], graph_overwrite]
    
    num_times_gotten_entities: int
    confirmed_entities: Annotated[list[Document], graph_overwrite]
    key_entities: Annotated[list[str], graph_extend]

    

    splitted_sources: Annotated[dict[str, list[Document]], graph_overwrite]

    key_insights: Annotated[list[str], graph_overwrite]
    requested_more_info: bool

    final_prompt: ChatPromptTemplate
    final_answer: str
    scratchpad: str # TODO: REMOVE THIS

graph_builder = StateGraph(State)

# define nodes and edges
def reset_state_node(state: State):
    return {
        'num_times_done_sim_search': 0,
        'confirmed_chunks': [],
        'chunks_to_exclude_on_second_pass': [],
        'num_times_gotten_entities': 0,
        'filtered': [],
        'confirmed_entities': [],
        'key_entities': "__RESET__",
        'requested_more_info': False
    }

def router_node(state: State):
    _input = {'question': state['prompt'], 'summary': state['summary']}
    use_guidelines = router_chain.invoke(_input)
    return {'use_guidelines': use_guidelines['b']}
def router_edge(state: State):
    if state['use_guidelines']:
        return '__use_guidelines__'
    else:
        return '__no__'
    
def db_picker_node(state: State):
    return {'use_knowledge_graph': True}
def db_picker_edge(state: State):
    if state['use_knowledge_graph']:
        return '__use_graph__'
    else:
        return '__use_vector__'


from itertools import product
def similarity_search_node(state: State):
    if not state['requested_more_info']:
        filter_query = state['prompt']
        if state['num_times_done_sim_search'] == 0:
            query = ''
            confirmed_chunks = []
            with ThreadPoolExecutor() as executor:
                collected = list(executor.map(
                    lambda guideline: db_retriever.get_chunks_by_similarity(
                        query=state['prompt'],
                        doc_ids_to_exclude=[],
                        source_guideline=guideline,
                        top_k_chunks=TOP_K_CHUNKS_PER_SIM_SEARCH_ON_BLIND_PASS
                    ),
                    GUIDELINE_NAMES
                ))
        elif state['num_times_done_sim_search'] == 1:
            doc_titles = [d.metadata[list(d.metadata.keys())[-1]] for d in state['filtered']]
            query = ', '.join(doc_titles)
            confirmed_chunks = state['filtered']
            with ThreadPoolExecutor() as executor:
                collected = list(executor.map(
                    lambda guideline: db_retriever.get_chunks_by_similarity(
                        query=query,
                        doc_ids_to_exclude=state['chunks_to_exclude_on_second_pass'],
                        source_guideline=guideline,
                        top_k_chunks=TOP_K_CHUNKS_PER_SIM_SEARCH_ON_SECOND_PASS
                    ),
                    GUIDELINE_NAMES
                ))
        chunks = reciprocal_rank_fusion(collected)
        chunk_ids = [chunk['id'] for chunk in chunks]
        chunks = [Document(page_content=chunk['chunkText'], metadata=chunk['metadata'])
                    for chunk in chunks]
        payloads = [FPayload(to_filter=chunks, filter_query=filter_query, extract_entities=False)]
    else: # requested more info (from node insights_per_guideline)
        confirmed_chunks = state['confirmed_chunks']
        queries = state['key_entities']

        with ThreadPoolExecutor() as executor:
            combination = list(product(GUIDELINE_NAMES, queries))
            collected = list(executor.map(
                lambda combo: db_retriever.get_chunks_by_similarity(
                    query=combo[1],
                    doc_ids_to_exclude=state['chunks_to_exclude_on_second_pass'],
                    source_guideline=combo[0],
                    top_k_chunks=TOP_K_CHUNKS_PER_SIM_SEARCH_ON_SECOND_PASS
                ),
                combination
            ))
        payloads = [FPayload(
                        to_filter=[Document(page_content=chunk['chunkText'], 
                                            metadata=chunk['metadata']
                                            ) 
                                            for chunk in chunks], 
                        filter_query=combo[1], 
                        extract_entities=False
                        ) 
                        for combo, chunks in zip(combination, collected)
                    ]
        chunk_ids = [chunk['id'] for chunks in collected for chunk in chunks]
    return {'fpayloads': payloads,
            'num_times_done_sim_search': state['num_times_done_sim_search'] + 1,
            'chunks_to_exclude_on_second_pass': chunk_ids,
            'confirmed_chunks': confirmed_chunks}

def filter_context_edge(state: State):
    if state['use_knowledge_graph']:
        return '__done__'
    if state['num_times_done_sim_search'] <= 1:
        return '__search_more__'
    else:
        return '__done__'    

def clarification_node(state: State):
    system = "You are part of an AI system that helps answer questions about Benign Prostate Hyperplasia (BPH)." + \
             "The node before you has determined that the user's query is unclear or unrelated to BPH. Either refuse to answer or politely ask for clarification."
    prompt = ChatPromptTemplate.from_messages([
        ('system', system),
        ('human', state['prompt'])
    ])
    chain = prompt | CLARIFYLLM | StrOutputParser()
    return {'final_answer': chain.invoke({})}
    
def multiquery_node(state: State):
    return {'multiqueries': [state['prompt']]}
    
def entity_retrieval_node(state: State):
    if not state['requested_more_info']:
        if state['num_times_gotten_entities'] == 0:
            confirmed_entities = []
            queries: list = state['multiqueries']
            entities = db_retriever.retrieve_entities(
                queries=queries, 
                top_entities_per_query=TOP_ENTITIES_PER_QUERY_ON_BLIND_PASS
            )
            final_queries = ''
        elif state['num_times_gotten_entities'] == 1:
            confirmed_entities: list[Document] = state['filtered']
            related = db_retriever.get_relationships_from_entities(entity_names=[e.metadata['entity_name'] for e in confirmed_entities])
            related_entities = [i['targetEntity'] for i in related['outside_rels']]
            query_from_entities = [entity.metadata['entity_name'] for entity in confirmed_entities]

            with ThreadPoolExecutor() as executor:
                guideline_chunks = list(executor.map(
                    lambda guideline: db_retriever.get_chunks_by_similarity(
                        query=state['prompt'],
                        doc_ids_to_exclude=[],
                        source_guideline=guideline,
                        top_k_chunks=TOP_K_CHUNKS_FOR_ENTITY_SEARCH
                    ), GUIDELINE_NAMES))
            chunks = reciprocal_rank_fusion(guideline_chunks)
            query_from_chunks = ', '.join([chunk['metadata'][list(chunk['metadata'].keys())[-1]] for chunk in chunks])
            
            final_queries = query_from_entities + [query_from_chunks]

            entities = db_retriever.retrieve_entities(
                queries=final_queries, 
                top_entities_per_query=TOP_ENTITIES_PER_QUERY_ON_SECOND_PASS
            )
            entities = [e for e in entities if e['name'] not in [c.metadata['entity_name'] for c in confirmed_entities]]
            entities.extend(related_entities)
    else: # requested more info (from node insights_per_guideline)
        confirmed_entities = state['confirmed_entities']
        insights = state['key_insights']
        final_queries = [insight for guideline in insights for insight in guideline['topics_to_know_more_about']]
        entities = db_retriever.retrieve_entities(
            queries=final_queries, 
            top_entities_per_query=TOP_ENTITIES_PER_QUERY_ON_SECOND_PASS
        )
        entities = [e for e in entities if e['name'] not in [c.metadata['entity_name'] for c in confirmed_entities]]

    entities = [Document(page_content=entity['descriptionText'], metadata={'entity_name': entity['name']})
                for entity in entities]
    return {'fpayloads': [FPayload(to_filter=e, filter_query=state['prompt'], extract_entities=False) for e in entities],
            'confirmed_entities': confirmed_entities,
            'num_times_gotten_entities': state['num_times_gotten_entities'] + 1,
            'scratchpad': final_queries}

"""def filter_node(state: State):  
    # TODO: generic filter node... does it need to be more task specific?
    payloads = state['filter_payloads']

    with ThreadPoolExecutor() as executor:
        collected = list(executor.map(
            lambda p: batch_doc_filter(
                filter_query=p['filter_query'], 
                documents=p['to_filter'],
                filter_on_metadata=FILTER_ON_METADATA,
                extract_entities=p['extract_entities']), 
            payloads
        ))
    docs = [f['docs'] for f in collected if f['docs']]
    key_entities = [f['key_entities'] for f in collected if f['key_entities']]
    filtered = reciprocal_rank_fusion(docs)
    # TODO: change query to actual rephrased query
    return {'filtered': filtered, 'key_entities': key_entities}"""
async def go_to_filter_edge(state: State):
    return [Send("filter_entity", {'payload': payload}) for payload in state['fpayloads']]
async def filter_node(state: FilterState):
    filtered = await doc_filter(document=state['payload']['to_filter'], filter_query=state['payload']['filter_query'], extract_entities=state['payload']['extract_entities'],
                                filter_on_metadata=FILTER_ON_METADATA)
    return {'filtered': filtered}


def filter_entity_edge(state: State):
    if state['num_times_gotten_entities'] <= 1:
        return '__get_more_entities__'
    else:
        return '__get_context__'

def context_retrieval_node(state: State):
    filtered_entities = state['filtered']
    confirmed_entities = state['confirmed_entities']
    entities = confirmed_entities + filtered_entities
    if not state['requested_more_info']:
        entity_names_to_ret_for = [entity.metadata['entity_name'] for entity in entities]
    else:
        entity_names_to_ret_for = [entity.metadata['entity_name'] for entity in filtered_entities]

    with ThreadPoolExecutor() as executor:
        collected = list(executor.map(
            lambda guideline: db_retriever.get_chunks_from_entities(
                query=state['prompt'],
                source_guideline=guideline, 
                entity_names=entity_names_to_ret_for,
                doc_ids_to_exclude=state['chunks_to_exclude_on_second_pass']), 
            GUIDELINE_NAMES
        ))
    chunks = reciprocal_rank_fusion(collected)
    chunk_ids = [chunk['id'] for chunk in chunks] + state['chunks_to_exclude_on_second_pass']
    chunks = [Document(page_content=chunk['chunkText'], metadata=chunk['metadata']) 
            for chunk in chunks]
    return {'fpayloads': [FPayload(to_filter=chunks, filter_query=state['prompt'], extract_entities=False)],
            'confirmed_entities': entities,
            'chunks_to_exclude_on_second_pass': chunk_ids}







def split_by_source_and_reorder_node(state: State):
    num_clusters = math_ceil(len(state['filtered'] + state['confirmed_chunks']) / CLUSTER_SIZE)

    clustering_filter = EmbeddingsClusteringFilter(
        embeddings=OpenAIEmbeddings(model=LARGE_EMBD),
        num_clusters=num_clusters,
        num_closest=CLUSTER_SIZE,
    )

    if state['requested_more_info']:
        all_sources = state['filtered'] + state['confirmed_chunks']
        # it's important to reorder the most relevant items first
        # when going through a request for more information, the most relevant items are the ones that came through on the second pass
        # when going through a request for more information, we should not dilute the secondarily retrieved docs by merging them with the first pass
        # therefore we cluster them separately
        clustered_first_pass = clustering_filter.transform_documents(state['confirmed_chunks'])
        clustered_first_pass = [Document(page_content=statedoc.page_content, metadata=statedoc.metadata) for statedoc in clustered_first_pass]
        clustered_first_pass = [clustered_first_pass[i:i + CLUSTER_SIZE] for i in range(0, len(clustered_first_pass), CLUSTER_SIZE)]
        clustered_second_pass = clustering_filter.transform_documents(state['filtered'])
        clustered_second_pass = [Document(page_content=statedoc.page_content, metadata=statedoc.metadata) for statedoc in clustered_second_pass]
        clustered_second_pass = [clustered_second_pass[i:i + CLUSTER_SIZE] for i in range(0, len(clustered_second_pass), CLUSTER_SIZE)]
        clustered = clustered_second_pass + clustered_first_pass
    else:
        all_sources = state['confirmed_chunks'] + state['filtered']
        clustered = clustering_filter.transform_documents(all_sources)
        # when not going through a request for more information, the most relevant items are the initial retrieved ones.
        # and we can cluster the whole batch together
        clustered = [Document(page_content=statedoc.page_content, metadata=statedoc.metadata) for statedoc in clustered]
        clustered = [clustered[i:i + CLUSTER_SIZE] for i in range(0, len(clustered), CLUSTER_SIZE)]

    cua, aua, eau = [], [], []
    for doc in all_sources:
        if 'CUA' in doc.metadata['Title']: cua.append(doc)
        elif 'AUA' in doc.metadata['Title']: aua.append(doc)
        elif 'EAU' in doc.metadata['Title']: eau.append(doc)
    for source in [cua, aua, eau]:
        source = doc_reorder.transform_documents(source)
    return {'splitted_sources': {'CUA': cua, 'AUA': aua, 'EAU': eau},
            'confirmed_chunks': all_sources,
            'scratchpad': clustered}




class NeedMoreInfo(TypedDict):
    info_sufficient: bool
    topics_to_know_more_about: Annotated[Optional[list[str]], ..., 'If there is insufficient information, provide the entities/topics you would like to know more about']
def insights_per_guideline_node(state: State):
    if not state['requested_more_info']:
        template = \
        """The following are exerpts from the {guideline} guideline on BPH (Benign Prostate Hyperplasia). 

        Is there enough information here to answer the patient's question? 
        We should ensure the key entities that the user is interested in or that would be pertinent to them are addressed
        But if there is enough information for a patient level answer, we can proceed to the next step without going for excruciating detail.
        
        <guideline_exerpts>
        {sources}
        </guideline_exerpts>"""
        prompt = ChatPromptTemplate.from_messages([
            ('system', template),
            ('human', "Query: "+state['prompt'])
        ])
        chain = prompt | CLARIFYLLM.with_structured_output(NeedMoreInfo, method='json_schema', strict=True)
        sources = [{k: '\n\n'.join([f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(v)])} for k, v in state['splitted_sources'].items()]
        ret = chain.batch([{'guideline': k, 'sources': v} for s in sources for k, v in s.items()])

        need_more_info = any([t for t in ret if not t['info_sufficient']])

        return {'key_insights': ret, 'requested_more_info': need_more_info}
    else:
        if debug := False:
            template = \
            """The following are exerpts from the {guideline} guideline on BPH. 

            Is there enough information here to answer the patient's question? 
            We should ensure the key entities that the user is interested in or that would be pertinent to them are addressed
            But if there is enough information for a patient level answer, we can proceed to the next step without going for excruciating detail.
            
            <guideline_exerpts>
            {sources}
            </guideline_exerpts>"""
            prompt = ChatPromptTemplate.from_messages([
                ('system', template),
                ('human', "Query: "+state['prompt']),
                ('ai', "{insight}"),
                ('human', "What about now, with these new sources?\n\n{new_sources} give me a clear rationale and answer for your reasoning.")
            ])
            old_sources = [s for s in state['confirmed_chunks'] if s not in state['filtered']]
            new_sources = [s for s in state['filtered']]
            old_cua, old_aua, old_eau = [], [], []
            new_cua, new_aua, new_eau = [], [], []
            for doc in old_sources:
                if 'CUA' in doc.metadata['Title']: old_cua.append(doc)
                elif 'AUA' in doc.metadata['Title']: old_aua.append(doc)
                elif 'EAU' in doc.metadata['Title']: old_eau.append(doc)
            for doc in new_sources:
                if 'CUA' in doc.metadata['Title']: new_cua.append(doc)
                elif 'AUA' in doc.metadata['Title']: new_aua.append(doc)
                elif 'EAU' in doc.metadata['Title']: new_eau.append(doc)
            old_sources = {'CUA': old_cua, 'AUA': old_aua, 'EAU': old_eau}
            new_sources = {'CUA': new_cua, 'AUA': new_aua, 'EAU': new_eau}
            old_sources = [{k: '\n\n'.join([f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(v)])} for k, v in old_sources.items()]
            new_sources = [{k: '\n\n'.join([f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(v)])} for k, v in new_sources.items()]
            chain = prompt | CLARIFYLLM.with_structured_output(NeedMoreInfo, method='json_schema', strict=True)

            ret = chain.batch([{'guideline': g, 'sources': v_s, 'new_sources': n_s, 'insight': str(i)} for g, v_s, n_s, i in zip(GUIDELINE_NAMES, old_sources, new_sources, state['key_insights'])])
            return {'requested_more_info': False, 'key_insights': ret}
        return {'requested_more_info': False}
def insights_edge(state: State):
    if state['requested_more_info']:
        return '__get_more_info__'
    else:
        return '__done__'
    


class InfoRequest(TypedDict):
    questions: Annotated[list[str], ..., 'List of new questions']
def collate_insights_node(state: State):
    template = \
    """Here is the scenario: multiple agents have each evaluated a set of documents, and have determined the key questions that need to be answered in order to respond to this user query:
    "{prompt}"


    There may or may not be duplicate questions across agents, each worded slightly differently.
    Your task is to return a new set of questions; your primary goal is to merge and deduplicate the questions from all agents.

    Try to keep the questions concise and clear, but do not remove any concepts that were present in the original questions.
    Try to return 3-6 questions, but you may return more if necessary.
    
    You must not make any inferences or assumptions. You must not add any additional questions.
    
    {questions}"""
    questions = ['\n'.join(info_request['topics_to_know_more_about']) for info_request in state['key_insights']]
    questions = [f'```<AGENT_#{i+1}>\n{questions[i]}\n</AGENT_#{i+1}>```' for i in range(len(questions))]
    questions = '\n\n'.join(questions)
    prompt = ChatPromptTemplate.from_messages([
        ('system', template)
    ])
    chain = prompt | CLARIFYLLM.with_structured_output(InfoRequest, method='json_schema', strict=True)
    ret = chain.invoke({'questions': questions, 'prompt': state['prompt']})
    return {'key_entities': ret['questions']}


def key_insights_by_source_node(state: State):
    pass

def context_compresion_node(state: State):
    template = \
    """You have been given documents that all roughly relate to one another within the realm of BPH.
    
    Extract key elements from each document as they relate to the key factors within the following example question:
    "{prompt}"

    You must only return your summarized key findings. 
    Do not make additional inferences or assumptions or commentary.
    Do not make any medical advice yourself, only provide the facts as they relate to the question.

    You must end your summary with key recommendations MADE BY THE GUIDELINES (CUA, AUA, EAU) that are relevant to the patient's question. You MUST NOT make any recommendations yourself.

    You can roughly structure your response as follows:
    1. ...
    2. ...
    3. ...
    etc.
    n. <guideline_name> Recommendations: ...

    Group your summaries by ENTITY, not by source document.
    
    <documents>
    {sources}
    </documents>"""
    sources = [[f'```<document_#{i+1}>\nSOURCE: {d.metadata.get("Title", d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(state['scratchpad'][i])] for i in range(len(state['scratchpad']))]
    prompt = ChatPromptTemplate.from_messages([
        ('system', template),
    ])
    chain = prompt | CLARIFYLLM | StrOutputParser()
    ret = chain.batch([{'sources': s, 'prompt': state['prompt']} for s in sources])
    ret = [Document(page_content=r, metadata={}) for r in ret]
    return {'confirmed_chunks': ret}

def outline_node(state: State):
    return
    sources = [f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(state['filtered'])]
    sources = '\n\n'.join(sources)
    key_topics = ', '.join(state['key_entities'])
    ret = outline_chain.invoke({'sources': sources, 'query': state['prompt'], 'key_topics': key_topics})
    return {'scratchpad': ret}#['scratchpad']}

def prompt_synthesis_node(state: State):
    sources = [f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' 
               for i, d in enumerate(state['confirmed_chunks'])]
    sources = '\n\n'.join(sources)
    final_prompt = synthesize_prompt(context=sources, outline=state['scratchpad'], memory=state['summary'], query=state['prompt'], rephrased=state['prompt'])
    # todo: rephrased query
    return {'final_prompt': final_prompt}

def chat_node(state: State):
    chat_chain = state['final_prompt'] | chat_llm_w_scratchpad 
    ret = chat_chain.invoke({})
    ret = ret.content
    return {'final_answer': ret}#['final_answer'], 'scratchpad': ret['scratchpad']}  # TODO: remove scratchpad

# --- add nodes ---
graph_builder.add_node('reset_state', reset_state_node)
graph_builder.add_node('router', router_node)
graph_builder.add_node('clarification', clarification_node)
graph_builder.add_node('multiquery', multiquery_node)
graph_builder.add_node('db_picker', db_picker_node)
graph_builder.add_node('similarity_search', similarity_search_node)
graph_builder.add_node('entity_retriever', entity_retrieval_node)
graph_builder.add_node('filter_entity', filter_node)
graph_builder.add_node('context_retriever', context_retrieval_node)
graph_builder.add_node('filter_context', filter_node)
graph_builder.add_node('split_by_source_and_reorder', split_by_source_and_reorder_node)
graph_builder.add_node('insights_per_guideline', insights_per_guideline_node)
graph_builder.add_node('collate_insights', collate_insights_node)
graph_builder.add_node('context_compresion', context_compresion_node)
graph_builder.add_node('prompt_synthesis', prompt_synthesis_node)
graph_builder.add_node('outline', outline_node)
graph_builder.add_node('chat', chat_node)

# --- add edges ---
graph_builder.add_edge(START, 'reset_state')
graph_builder.add_edge('reset_state', 'router')
graph_builder.add_conditional_edges(
    source='router',
    path=router_edge,
    path_map={'__use_guidelines__': 'db_picker', '__no__': 'clarification'})
graph_builder.add_conditional_edges(
    source='db_picker',
    path=db_picker_edge,
    path_map={'__use_vector__': 'similarity_search', '__use_graph__': 'multiquery'})

graph_builder.add_edge('clarification', END)
graph_builder.add_edge('similarity_search', 'filter_context')

graph_builder.add_edge('multiquery', 'entity_retriever')
graph_builder.add_conditional_edges('entity_retriever', go_to_filter_edge, ['filter_entity'])
graph_builder.add_conditional_edges(
    source='filter_entity',
    path=filter_entity_edge,
    path_map={'__get_more_entities__': 'entity_retriever', '__get_context__': 'context_retriever'})
graph_builder.add_edge('context_retriever', 'filter_context')

graph_builder.add_conditional_edges(
    source='filter_context',
    path=filter_context_edge,
    path_map={'__search_more__': 'similarity_search', '__done__': 'split_by_source_and_reorder'}
)

graph_builder.add_edge('split_by_source_and_reorder', 'insights_per_guideline')

graph_builder.add_conditional_edges(
    source='insights_per_guideline',
    path=insights_edge,
    path_map={'__get_more_info__': 'collate_insights', '__done__': 'context_compresion'}
)
graph_builder.add_edge('collate_insights', 'similarity_search')

graph_builder.add_edge('context_compresion', 'outline')
graph_builder.add_edge('outline', 'prompt_synthesis')
graph_builder.add_edge('prompt_synthesis', 'chat')
graph_builder.add_edge('chat', END)

# --- compile graph ---
graph = graph_builder.compile()