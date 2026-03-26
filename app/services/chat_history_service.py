from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter

message_store = InMemoryChatMessageStore()


def get_message_store():
    return message_store


def get_message_retriever():
    return ChatMessageRetriever(message_store)


def get_message_writer():
    return ChatMessageWriter(message_store)