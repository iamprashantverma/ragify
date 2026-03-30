from haystack.components.preprocessors import DocumentSplitter

def get_document_splitter():
    return DocumentSplitter(
        split_by="sentence",     
        split_length=10, 
        split_overlap=2  
    )