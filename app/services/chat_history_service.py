from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

chat_message_store = InMemoryChatMessageStore()

def get_chat_store():
    return chat_message_store
