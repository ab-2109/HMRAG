from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.web_retrieval import WebRetrieval
from agents.summary_agent import SummaryAgent
from agents.decompose_agent import DecomposeAgent

from typing import List
import json

class MRetrievalAgent():
    def __init__(self, config):
        self.config = config
        self.vector_retrieval = VectorRetrieval(config)
        self.graph_retrieval = GraphRetrieval(config)
        self.web_retrieval = WebRetrieval(config)
        self.sum_agent = SummaryAgent(config)
        self.dec_agent = DecomposeAgent(config)
        
    def predict(self, problems, shot_qids, qid):
        problem = problems[qid]
        question = problem['question']
        
        # Decompose can return a list of sub-queries or a single query string
        sub_queries = self.dec_agent.decompose(question)
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]
        
        # Collect retrieval results for all sub-queries
        all_vector_responses = []
        all_graph_responses = []
        all_web_responses = []
        
        for sub_q in sub_queries:
            vector_response = self.vector_retrieval.find_top_k(sub_q)
            graph_response = self.graph_retrieval.find_top_k(sub_q)
            web_response = self.web_retrieval.find_top_k(sub_q)
            all_vector_responses.append(str(vector_response))
            all_graph_responses.append(str(graph_response))
            all_web_responses.append(str(web_response))
        
        vector_combined = "\n".join(all_vector_responses)
        graph_combined = "\n".join(all_graph_responses)
        web_combined = "\n".join(all_web_responses)

        all_messages = [
            "Vector Retrieval Agent:\n" + vector_combined + "\n", 
            "Graph Retrieval Agent:\n" + graph_combined + "\n",
            "Web Retrieval Agent:\n" + web_combined + "\n"
        ]
        
        final_ans, final_messages = self.sum(problems, shot_qids, qid, all_messages)
        return final_ans, final_messages
        
        
    def sum(self, problems, shot_qids, qid, sum_question):
        final_ans, all_messages = self.sum_agent.summarize(problems, shot_qids, qid, sum_question)
        return final_ans, all_messages
