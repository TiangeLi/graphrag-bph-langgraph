from dotenv import load_dotenv
load_dotenv('.env', override=True)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from ast import literal_eval
from typing import TypedDict

from helpers.constants import CLARIFYLLM

from nodes.router import router_chain
from nodes.db_retriever import DBRetriever
from nodes.doc_filter import batch_doc_filter
from nodes.prompt_synthesizer import synthesize_prompt
from nodes.chat import chat_llm_w_scratchpad

# --- utilities ---
db_retriever = DBRetriever(index_name='entity', chunks=True, entities=True)

# --- basic graph elements ---
class State(TypedDict):
    prompt: str
    use_guidelines: bool
    summary: str
    raw_ret_chunks: list
    filtered_chunks: list
    final_prompt: ChatPromptTemplate
    final_answer: str
    scratchpad: str # TODO: REMOVE THIS
    debug: str # TODO: REMOVE THIS
graph_builder = StateGraph(State)

# define nodes and edges
def router_node(state: State):
    _input = {'question': state['prompt'], 'summary': state['summary']}
    use_guidelines = router_chain.invoke(_input)
    return {'use_guidelines': use_guidelines['b']}
def router_edge(state: State):
    if state['use_guidelines']:
        return '__use_guidelines__'
    else:
        return '__no__'

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
    pass # TODO: expand this to do multi query later
    
def db_retriever_node(state: State):
    queries: list = [state['prompt']]  # TODO: replace with multiquery
    retrieved = db_retriever.retrieve(queries)
    chunks = [Document(page_content=chunk['chunkText'], metadata=literal_eval(chunk['metadata'])) 
              for chunk in retrieved['doc_chunks']]
    #communities = retrieved['community_reports']
    # ... function to repackage communities data
    #relationships = retrieved['entity_relationships']
    # ... function to repackage relationships data
    entities = retrieved['entities']
    # ... function to repackage entities data
    return {'raw_ret_chunks': chunks, 'len_raw_ret_chunks': len(chunks), "debug": [entities['name'] for entities in entities]}

def filter_node(state: State):
    filtered_retrieved = batch_doc_filter(rephrased_query=state['prompt'], documents=state['raw_ret_chunks'])
    # TODO: change query to actual rephrased query
    return {'filtered_chunks': filtered_retrieved, 'len_filtered_chunks': len(filtered_retrieved)}

def prompt_synthesis_node(state: State):
    sources = [f'```<document_#{i+1}>\nSOURCE: {d.metadata.get('Title', d.metadata)}\n\n{d.page_content}\n</document_#{i+1}>```' for i, d in enumerate(state['filtered_chunks'])]
    sources = '\n\n'.join(sources)
    final_prompt = synthesize_prompt(context=sources, memory=state['summary'], query=state['prompt'], rephrased=state['prompt'])
    # todo: rephrased query
    return {'final_prompt': final_prompt}

def chat_node(state: State):
    chat_chain = state['final_prompt'] | chat_llm_w_scratchpad 
    ret = chat_chain.invoke({})
    return {'final_answer': ret['final_answer'], 'scratchpad': ret['scratchpad']}  # TODO: remove scratchpad

# --- add nodes ---
graph_builder.add_node('router', router_node)
graph_builder.add_node('clarification', clarification_node)
graph_builder.add_node('multiquery', multiquery_node)
graph_builder.add_node('db_retriever', db_retriever_node)
graph_builder.add_node('filter', filter_node)
graph_builder.add_node('prompt_synthesis', prompt_synthesis_node)
graph_builder.add_node('chat', chat_node)

# --- add edges ---
graph_builder.add_edge(START, 'router')
graph_builder.add_conditional_edges(
    source='router',
    path=router_edge,
    path_map={'__use_guidelines__': 'multiquery', '__no__': 'clarification'})
graph_builder.add_edge('clarification', END)
graph_builder.add_edge('multiquery', 'db_retriever')
graph_builder.add_edge('db_retriever', 'filter')
graph_builder.add_edge('filter', 'prompt_synthesis')
graph_builder.add_edge('prompt_synthesis', 'chat')
graph_builder.add_edge('chat', END)

# --- compile graph ---
graph = graph_builder.compile()