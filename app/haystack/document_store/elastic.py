from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

def get_document_store():
    return ElasticsearchDocumentStore(hosts="http://localhost:9200")

document_store = get_document_store()