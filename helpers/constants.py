from langchain_openai import ChatOpenAI

#EDUCATION_LEVEL = 'Your answer should be highly sophisticated, at the level of a post doctoral researcher in the field.'
EDUCATION_LEVEL = 'Your answer should be succinct and patient oriented, at a grade 12 reading level.'

# ------------------------------------------------------------------- #
# when filtering, should we do a double filter using metadata --> main text, or just main text?
FILTER_ON_METADATA = True

# ------------------------------------------------------------------- #
CLUSTER_SIZE = 20
# at the end of the pipeline, we make clusters by similarity so that we don't have so many docs per llm call

# ------------------------------------------------------------------- #
# Basic similarity search retrieval parameters
TOP_K_CHUNKS_PER_SIM_SEARCH_ON_BLIND_PASS = 5
TOP_K_CHUNKS_PER_SIM_SEARCH_ON_SECOND_PASS = 3
# graph vector retrieval parameters
TOP_ENTITIES_PER_QUERY_ON_BLIND_PASS = 6 # i.e. dynamic based on size of multiquery
TOP_ENTITIES_PER_QUERY_ON_SECOND_PASS = 3  # second pass is retrieved based on entities retreived from first pass
TOP_K_CHUNKS_PER_ENTITY = 3
TOP_K_CHUNKS_FOR_ENTITY_SEARCH = 3 # do a similarity search and embed those names for entity search
TOP_COMMUNITIES = 1
TOP_OUTSIDE_RELS = 2
TOP_INSIDE_RELS = 1

# ------------------------------------------------------------------- #
# OpenAI model parameters

LARGE_EMBD = 'text-embedding-3-large'
SMALL_EMBD = 'text-embedding-3-small'

BIG_MODEL = 'gpt-4o-2024-08-06'
SMALL_MODEL = 'gpt-4o-mini-2024-07-18'

EXPERIMENTAL_LATEST_CHAT_MODEL = 'chatgpt-4o-latest'

# we go in such painful granularity on the models,
# to future proof easy model swapping on each function
# Obviously doesn't matter right now

# MAIN
CONVLLM = ChatOpenAI(model=EXPERIMENTAL_LATEST_CHAT_MODEL, temperature=0.2, streaming=True)

# Router
ROUTERLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)
CLARIFYLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0.6, streaming=True)

# Doc Filter
FILTLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0, streaming=True)

# Outliner
OUTLINELLM = ChatOpenAI(model=SMALL_MODEL, temperature=0.3, streaming=True)