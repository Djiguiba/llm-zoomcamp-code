INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class RAGBase:
    def __init__(self,
                 index,
                 llm_client,
                 instructions=INSTRUCTIONS,
                 prompt_template=PROMPT_TEMPLATE,
                 course="llm-zoomcamp",
                 model="gpt-5.4-mini"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def elastic_search(self, query, index_name="course-questions", num_results=5):
        search_query = {
            "size": num_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": self.course
                        }
                    }
                }
            }
        }

        response = self.index.search(
            index=index_name,
            body=search_query
        )

        return [doc['_source'] for doc in response['hits']['hits']]
                
        

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append("Q :" + doc['question'])
            lines.append("A :" + doc['answer'])
            lines.append("")

        return "\n".join(lines).strip()


    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query,
            context=context
        )

    def llm(self, prompt):
        input_message = [
            #{"role": "developer", "content": self.instructions}, # Use this role if you use openai client
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.messages.create( # replace messages method by responses if you use openai client 
            model=self.model,
            system=self.instructions,
            max_tokens=1000,
            messages=input_message # replace messages attribut by input if you use openai client
        )
        return response

    def rag(self, query):
        search_results = self.search(query)
        #search_results = self.elastic_search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        return answer


class RAGVector(RAGBase):
    def __init__(self, embedder, **kargs):
        super().__init__(**kargs)
        self.embedder = embedder


    def search(self, query, num_results=5):
        filter_dict = {"course": self.course}
        query_vector = self.embedder.encode(query)

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )


class RAGPgVector(RAGBase):
    def __init__(self, embedder, conn, **kargs):
        super().__init__(index=None, **kargs)
        self.embedder = embedder
        self.conn = conn

    def vec_to_str(self, vector):
        return "[" + ",".join(str(x) for x in vector) + "]"

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = self.vec_to_str(query_vector)

        results = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results) 
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in results
        ]

    
        