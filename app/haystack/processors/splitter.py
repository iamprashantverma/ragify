from haystack.components.preprocessors import DocumentSplitter

def get_document_splitter(split_by: str = "word", split_length: int = 150, split_overlap: int = 20):
    return DocumentSplitter(
        split_by=split_by,
        split_length=split_length,
        split_overlap=split_overlap
    )

