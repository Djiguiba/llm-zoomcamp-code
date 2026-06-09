""" Loading Data and Building the search index """

import requests
from minsearch import Index
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm


def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"
    
    for course in courses_raw:
        course_url = f"""{url_prefix}{course['path']}"""
    
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()
    
        documents.extend(course_data)

    return documents


def build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index


def build_elastic_index(documents):
    index_settings = index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                #"text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"} 
            }
        }
    }
    index_name = "course-questions"
    
    elastic_client = Elasticsearch("http://localhost:9200")
    elastic_client.indices.create(
        index=index_name,
        body=index_settings
    )

    for doc in tqdm(documents):
        elastic_client.index(
            index=index_name,
            document=doc
        )

    return elastic_client
    
    