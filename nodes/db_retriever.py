from os import getenv

from langchain_openai import OpenAIEmbeddings

from neo4j import GraphDatabase, Result
from typing import Dict, Any
from pydantic import BaseModel
from ast import literal_eval
from json import loads as jsonloads
from concurrent.futures import ThreadPoolExecutor

from helpers.utils import reciprocal_rank_fusion
from helpers.custom_lcneo4j import Neo4jVector as CustomNeo4jVector
from helpers.constants import (
    LARGE_EMBD,
    TOP_K_CHUNKS_PER_ENTITY,
    TOP_COMMUNITIES,
    TOP_OUTSIDE_RELS,
    TOP_INSIDE_RELS
)

NEO4J_URI=getenv("NEO4J_URI")
NEO4J_USERNAME=getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=getenv("NEO4J_PASSWORD")


# note:
# we had to change up the LangChain Neo4jVector class to not use dict to YAML str; this is a temporary fix. 
# check langchain_community.vectorstores frequently for updates to Neo4jVector

class VectorIndex(BaseModel):
    index_name: str
    embedding_dims: int
    similarity_function: str
    embd_model: str
    embd_node_prop: str

class DBRetriever(object):
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.descriptions_vector = VectorIndex(index_name='description_vector', 
                                               embedding_dims=3072,
                                               similarity_function='cosine',
                                               embd_model=LARGE_EMBD,
                                               embd_node_prop='description_embedding')
        self.chunks_vector = VectorIndex(index_name='chunks_vector',
                                            embedding_dims=3072, 
                                            similarity_function='cosine',
                                            embd_model=LARGE_EMBD,
                                            embd_node_prop='chunk_embedding_3072')

        self.setup_vector_index()
        self.kb_desc_retriever = self.get_kb_retriever(self.descriptions_vector)
        self.chunks_sim_retriever = self.get_vector_retreiver()
        self.chunks_embedder = OpenAIEmbeddings(model=self.chunks_vector.embd_model)

    def get_chunks_by_similarity(self, query: str,
                                 doc_ids_to_exclude: list[str],
                                 source_guideline: str,
                                 top_k_chunks: int):
        ret = self.chunks_sim_retriever.similarity_search(
            query=query, k=top_k_chunks,
            filter={"metadata": {"$like": source_guideline.upper()},
                    "id": {"$nin": doc_ids_to_exclude}
                    })
        return [{'chunkText': c.page_content,
                 'metadata': literal_eval(c.metadata['metadata']),
                 'id': c.metadata['id']}
                 for c in ret]

    def retrieve_entities(self, queries: list, top_entities_per_query: int):
        def _search_single_query(query):
            ret = self.kb_desc_retriever.similarity_search(query=query, k=top_entities_per_query)
            entities = jsonloads(ret[0].page_content)['entities']
            return entities
        with ThreadPoolExecutor() as executor:
            all_entities = list(executor.map(_search_single_query, queries))
        all_entities = reciprocal_rank_fusion(all_entities)
        return all_entities
    
    def get_vector_retreiver(self):
        # Returns just the regular vector retriever on raw documents for good ol similarity search
        retrieval_query = """
            MATCH (node)
            RETURN node.text AS text, 1.0 AS score, {
                metadata: node.metadata,
                id: node.id
            } AS metadata
        """
        return CustomNeo4jVector.from_existing_index(
            embedding=OpenAIEmbeddings(model=LARGE_EMBD),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="chunks_vector",    
            retrieval_query=retrieval_query
        )

    def get_kb_retriever(self, vector_store: VectorIndex):
        retrieval_query = """
            WITH collect(node) as nodes
            WITH
            collect {
                UNWIND nodes as n
                RETURN {name: n.name, descriptionText: n.description} AS entityData
            } as collected
            RETURN {
                entities: collected
            } AS text, 
            1.0 AS score, 
            {} AS metadata
        """
        return CustomNeo4jVector.from_existing_index(
            embedding=OpenAIEmbeddings(model=vector_store.embd_model),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=vector_store.index_name,
            retrieval_query=retrieval_query,
        )
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """Executes a Cypher statement"""
        return self.driver.execute_query(cypher, parameters_=params, result_transformer_=Result.to_df)
    
    def setup_vector_index(self):
        """If not created, this sets up the vector indices, then calculates and stores the community weights"""
        self.db_query(
            """
                CREATE VECTOR INDEX $index_name IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $embedding_dims,
                    `vector.similarity_function`: $similarity_function}}
            """,
            params={
                'index_name': self.descriptions_vector.index_name,
                'embedding_dims': self.descriptions_vector.embedding_dims,
                'similarity_function': self.descriptions_vector.similarity_function
            })
        self.db_query(
            """
                MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
                WITH n, count(distinct c) AS chunkCount
                SET n.weight = chunkCount
            """
        )

    def get_chunks_from_entities(self, 
                                 query: str,
                                 source_guideline: str,
                                 entity_names: list[str], 
                                 doc_ids_to_exclude: list[str],
                                 top_k_chunks_per_entity: int = TOP_K_CHUNKS_PER_ENTITY):
        query_embd = self.chunks_embedder.embed_query(query)
        top_k_chunks = len(entity_names) * top_k_chunks_per_entity
        cypher = f"""
            UNWIND $entity_names AS entity_name
            MATCH (e:__Entity__ {{name: entity_name}})<-[:HAS_ENTITY]->(c:__Chunk__)
            WHERE c.metadata CONTAINS $source_guideline AND NOT c.id IN $doc_ids_to_exclude
            WITH c, count(distinct e) as freq, vector.similarity.cosine(
                c.{self.chunks_vector.embd_node_prop}, $embedded) as similarity
            ORDER BY freq DESC, similarity DESC
            LIMIT $top_k_chunks
            RETURN c.text AS chunkText, c.metadata AS metadata, c.id AS id
        """
        result = self.db_query(cypher, params={
            'entity_names': entity_names, 
            'top_k_chunks': top_k_chunks,
            'source_guideline': source_guideline.upper(),
            'embedded': query_embd,
            'doc_ids_to_exclude': doc_ids_to_exclude
        })
        return [{'chunkText': record['chunkText'], 
                 'metadata': literal_eval(record['metadata']),
                 'id': record['id']} 
                 for _, record in result.iterrows()]

    def get_communities_from_entities(self, entity_names: list[str], top_communities: int = TOP_COMMUNITIES):
        cypher = """
            UNWIND $entity_names AS entity_name
            MATCH (e:__Entity__ {name: entity_name})-[:IN_COMMUNITY]->(c:__Community__)
            WITH c, c.rank as rank, c.weight AS weight
            ORDER BY rank, weight DESC
            LIMIT $top_communities
            // RETURN c.title AS name, c.summary AS summary, rank, weight
            RETURN c.title AS name, c.full_content AS content, rank, weight
        """
        result = self.db_query(cypher, params={
            'entity_names': entity_names, 
            'top_communities': top_communities
        })
        return [{'name': record['name'],
                 'content': record['content'], 
                 'rank': record['rank'], 
                 'weight': record['weight']} 
                 for _, record in result.iterrows()]
    
    def get_relationships_from_entities(self, entity_names: list[str], 
                                        top_inside_rels: int = TOP_INSIDE_RELS,
                                        top_outside_rels: int = TOP_OUTSIDE_RELS):
        cypher = """
            UNWIND $entity_names AS entity_name
            MATCH (e:__Entity__ {name: entity_name})-[r:RELATED]-(m)
            WITH e, r, m,
                CASE WHEN m.name IN $entity_names THEN 'inside' ELSE 'outside' END AS rel_type
            WITH rel_type, {
                sourceEntity: e.name,
                targetEntity: {name: m.name, descriptionText: m.description},
                descriptionText: r.description,
                rank: r.rank,
                weight: r.weight
            } AS relationshipData
            ORDER BY relationshipData.rank, relationshipData.weight DESC
            WITH rel_type, collect(relationshipData) AS relationships
            RETURN {
                insideRels: [rel IN relationships WHERE rel_type = 'inside'][..$top_inside_rels],
                outsideRels: [rel IN relationships WHERE rel_type = 'outside'][..$top_outside_rels]
            } AS result
        """
        result = self.db_query(cypher, params={
            'entity_names': entity_names, 
            'top_inside_rels': top_inside_rels,
            'top_outside_rels': top_outside_rels
        })
        inside = []
        outside = []
        for _, record in result.iterrows():
            inside.extend(record['result']['insideRels'])
            outside.extend(record['result']['outsideRels'])
        return {'inside_rels': inside, 'outside_rels': outside}