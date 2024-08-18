from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

#EDUCATION_LEVEL = 'Your answer should be highly sophisticated, at the level of a post doctoral researcher in the field.'
EDUCATION_LEVEL = 'Your answer should be succinct and patient oriented, at a grade 12 reading level.'

# ------------------------------------------------------------------- #
# when filtering, should we do a double filter using metadata --> main text, or just main text?
ADD_METADATA_FILTER = True

# ------------------------------------------------------------------- #
# Neo4j graph vector retrieval parameters
top_entities = 10
top_k_chunks = 30
top_communities = -1
top_outside_rels = -1
top_inside_rels = -1

# ------------------------------------------------------------------- #
# OpenAI model parameters

EMBD = OpenAIEmbeddings(model='text-embedding-3-large')

BIG_MODEL = 'gpt-4o-2024-08-06'
SMALL_MODEL = 'gpt-4o-mini-2024-07-18'

EXPERIMENTAL_LATEST_CHAT_MODEL = 'chatgpt-4o-latest'

# we go in such painful granularity on the models,
# to future proof easy model swapping on each function
# Obviously doesn't matter right now

# MAIN
CONVLLM = ChatOpenAI(model=BIG_MODEL, temperature=0.5, streaming=True)

# Router
ROUTERLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)
CLARIFYLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0.6, streaming=True)

# Doc Filter
FILTLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)