from os import getenv

from neo4j import GraphDatabase, Result
from typing import Dict, Any
from json import loads as jsonloads
from concurrent.futures import ThreadPoolExecutor

from helpers.utils import reciprocal_rank_fusion
from helpers.custom_lcneo4j import Neo4jVector as CustomNeo4jVector
from helpers.constants import EMBD, top_entities, top_k_chunks, top_communities, top_inside_rels, top_outside_rels

NEO4J_URI=getenv("NEO4J_URI")
NEO4J_USERNAME=getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=getenv("NEO4J_PASSWORD")


# note:
# we had to change up the LangChain Neo4jVector class to not use dict to YAML str; this is a temporary fix. 
# check langchain_community.vectorstores frequently for updates to Neo4jVector

class DBRetriever(object):
    def __init__(self, index_name='entity', **items_to_retrieve):
        """items_to_retrieve (kwargs):
            chunks (bool): Whether to include chunk information
            communities (bool): Whether to include community information
            relationships (bool): Whether to include relationship information
            entities (bool): Whether to include entity information"""
        self.index_name = index_name
        self.items_to_retrieve = items_to_retrieve
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embedding_dims = 3072
        self.similarity_function = 'cosine'

        self.setup_vector_index()
        self.retriever = self.setup_neo4j_retriever()

    def retrieve(self, queries: list):
        def _search_query(query):
            return self.retriever.similarity_search(
                query=query,
                k=top_entities,
                params={
                    'top_k_chunks': top_k_chunks,
                    'top_communities': top_communities,
                    'top_inside_rels': top_inside_rels,
                    'top_outside_rels': top_outside_rels
                }
            )
        with ThreadPoolExecutor() as executor:
            docs = list(executor.map(_search_query, queries))
        docs = [jsonloads(doc[0].page_content) for doc in docs]
        final_docs = {}
        items_map = {  # maps self.items_to_retrieve to the keys in the final_docs dict
            'chunks': 'doc_chunks',
            'communities': 'community_reports',
            'relationships': 'entity_relationships',
            'entities': 'entities'
        }
        for item_to_retrieve, doc_key in items_map.items():
            if self.items_to_retrieve.get(item_to_retrieve):
                final_docs[doc_key] = reciprocal_rank_fusion([doc.get(doc_key) for doc in docs])
        return final_docs
    
    def setup_neo4j_retriever(self):
        retrieval_query = self.create_retrieval_query()
        return CustomNeo4jVector.from_existing_index(
            embedding=EMBD,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=self.index_name,
            retrieval_query=retrieval_query,
        )
    
    def setup_vector_index(self):
        """If not created, this sets up the vector index self.index_name, then calculates and stores the community weights"""
        self.db_query(
            f"""
            CREATE VECTOR INDEX {self.index_name}"""
                + """ IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
            OPTIONS {indexConfig: {\n"""
                + f"""`vector.dimensions`: {self.embedding_dims},
            `vector.similarity_function`: '{self.similarity_function}'
            }}}}""")
        self.db_query(
            """
            MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
            WITH n, count(distinct c) AS chunkCount
            SET n.weight = chunkCount""")

    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """Executes a Cypher statement; returns a DataFrame if desired"""
        return self.driver.execute_query(cypher, parameters_=params, result_transformer_=Result.to_df)
    
    def create_retrieval_query(self):
        """Creates a Cypher query to retrieve information"""

        base_str = """
        WITH collect(node) as nodes"""
        ret_str = []

        if self.items_to_retrieve.get('chunks'):
            ret_str.append("doc_chunks: text_mapping")
            base_str += """
            // Entity - Text Unit Mapping
            WITH
            collect {
                UNWIND nodes as n
                MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
                WITH c, count(distinct n) as freq
                RETURN {chunkText: c.text, metadata: c.metadata, frequency: freq} AS chunkData
                ORDER BY freq DESC
                LIMIT $top_k_chunks
            } AS text_mapping"""

        if self.items_to_retrieve.get('communities'):
            ret_str.append("community_reports: report_mapping")
            base_str += """,
            // Entity - Report Mapping
            collect {
                UNWIND nodes as n
                MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
                WITH c, c.rank as rank, c.weight AS weight
                // RETURN {summary: c.summary, rank: rank, weight: weight} AS reportData
                RETURN {content: c.full_content, rank: rank, weight: weight} AS reportData
                ORDER BY rank, weight DESC
                LIMIT $top_communities
            } AS report_mapping"""

        if self.items_to_retrieve.get('relationships'):
            ret_str.append("entity_relationships: outsideRels + insideRels")
            base_str += """,
            // Outside Relationships 
            collect {
                UNWIND nodes as n
                MATCH (n)-[r:RELATED]-(m) 
                WHERE NOT m IN nodes
                RETURN {descriptionText: r.description, rank: r.rank, weight: r.weight} AS relationshipData
                ORDER BY r.rank, r.weight DESC 
                LIMIT $top_outside_rels
            } as outsideRels,
            // Inside Relationships 
            collect {
                UNWIND nodes as n
                MATCH (n)-[r:RELATED]-(m) 
                WHERE m IN nodes
                RETURN {descriptionText: r.description, rank: r.rank, weight: r.weight} AS relationshipData
                ORDER BY r.rank, r.weight DESC 
                LIMIT $top_inside_rels
            } as insideRels"""

        if self.items_to_retrieve.get('entities'):
            ret_str.append("entities: ret_entities")
            base_str += """,
            // Entities description
            collect {
                UNWIND nodes as n
                RETURN {name: n.name, descriptionText: n.description} AS entityData
            } as ret_entities"""

        final = base_str + '\n' +\
            "// Return as structured JSON\n" +\
            "RETURN {"+\
            f"{','.join(ret_str)}"+\
            """} AS text, 
            1.0 AS score, 
            {} AS metadata
            """
        return final