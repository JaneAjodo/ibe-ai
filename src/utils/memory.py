from typing import Dict, List
from src.models.schemas import ChatMessage
from src.core.config import settings
from src.core.logging import setup_logger
from datetime import datetime

logger = setup_logger("memory")

class ConversationMemory:
    def __init__(self):
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append(
            ChatMessage(role=role, content=content, timestamp=datetime.utcnow())
        )
        
        # Keep only last N messages per session
        max_msgs = settings.MAX_CONVERSATION_HISTORY * 2
        if len(self._sessions[session_id]) > max_msgs:
            self._sessions[session_id] = self._sessions[session_id][-max_msgs:]

    def get(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.get(session_id, [])

    def format_for_prompt(self, session_id: str) -> str:
        history = self.get(session_id)
        if not history:
            return ""
        lines = []
        for msg in history[-6:]:  # Last 3 exchanges
            prefix = "User" if msg.role == "user" else "Ibe"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_all_sessions(self) -> List[str]:
        return list(self._sessions.keys())

# Singleton
_memory = ConversationMemory()

def get_memory() -> ConversationMemory:
    return _memory
