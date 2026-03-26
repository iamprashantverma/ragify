from typing import List, Dict
from datetime import datetime

# In-memory storage for chat history (use database in production)
chat_history_store: Dict[str, List[Dict]] = {}

def add_to_history(user_id: str, query: str, response: str):
    """Add a conversation to user's chat history"""
    if user_id not in chat_history_store:
        chat_history_store[user_id] = []
    
    chat_history_store[user_id].append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    })
    
    # Keep only last 10 conversations to avoid memory issues
    if len(chat_history_store[user_id]) > 10:
        chat_history_store[user_id] = chat_history_store[user_id][-10:]

def get_history(user_id: str, limit: int = 5) -> List[Dict]:
    """Get user's recent chat history"""
    if user_id not in chat_history_store:
        return []
    
    return chat_history_store[user_id][-limit:]

def format_history_for_context(user_id: str, limit: int = 3) -> str:
    """Format chat history as context string"""
    history = get_history(user_id, limit)
    
    if not history:
        return ""
    
    context = "Previous conversation:\n"
    for item in history:
        context += f"User: {item['query']}\n"
        context += f"Assistant: {item['response'][:200]}...\n\n"  # Truncate long responses
    
    return context

def clear_history(user_id: str):
    """Clear user's chat history"""
    if user_id in chat_history_store:
        del chat_history_store[user_id]
