"""
Chat Session Manager for Research Paper RAG System

This module manages multiple chat sessions with persistent storage:
- Create new chat sessions
- Save and load chat history
- Resume previous conversations
- List all available chats
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ChatSession:
    """Represents a single chat session"""
    
    def __init__(self, session_id: str, paper_title: str = "Unknown", created_at: str = None):
        self.session_id = session_id
        self.paper_title = paper_title
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = self.created_at
        self.chat_history: List[tuple] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_exchange(self, question: str, answer: str):
        """Add a Q&A exchange to the history"""
        self.chat_history.append((question, answer))
        self.updated_at = datetime.now().isoformat()
    
    def get_history(self) -> List[tuple]:
        """Get the chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = []
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "paper_title": self.paper_title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "chat_history": self.chat_history,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create from dictionary"""
        session = cls(
            session_id=data["session_id"],
            paper_title=data.get("paper_title", "Unknown"),
            created_at=data.get("created_at")
        )
        session.updated_at = data.get("updated_at", session.created_at)
        session.chat_history = [tuple(item) for item in data.get("chat_history", [])]
        session.metadata = data.get("metadata", {})
        return session


class ChatManager:
    """Manages multiple chat sessions with persistent storage"""
    
    def __init__(self, storage_dir: str = "chat_sessions"):
        """
        Initialize chat manager
        
        Args:
            storage_dir: Directory to store chat sessions
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_session: Optional[ChatSession] = None
        logger.info(f"Chat manager initialized with storage at: {self.storage_dir}")
    
    def create_new_session(self, paper_title: str = "Unknown") -> ChatSession:
        """
        Create a new chat session
        
        Args:
            paper_title: Title of the paper for this session
            
        Returns:
            New ChatSession instance
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = ChatSession(session_id=session_id, paper_title=paper_title)
        self.current_session = session
        self.save_session(session)
        logger.info(f"Created new chat session: {session_id}")
        return session
    
    def save_session(self, session: ChatSession):
        """
        Save a chat session to disk
        
        Args:
            session: ChatSession to save
        """
        filepath = self.storage_dir / f"{session.session_id}.json"
        with open(filepath, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        logger.info(f"Saved chat session: {session.session_id}")
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Load a chat session from disk
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Loaded ChatSession or None if not found
        """
        filepath = self.storage_dir / f"{session_id}.json"
        
        if not filepath.exists():
            logger.error(f"Session not found: {session_id}")
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = ChatSession.from_dict(data)
        self.current_session = session
        logger.info(f"Loaded chat session: {session_id} with {len(session.chat_history)} messages")
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available chat sessions
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        for filepath in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Create summary
                created = datetime.fromisoformat(data["created_at"])
                updated = datetime.fromisoformat(data["updated_at"])
                
                sessions.append({
                    "session_id": data["session_id"],
                    "paper_title": data.get("paper_title", "Unknown"),
                    "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": updated.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_messages": len(data.get("chat_history", [])),
                    "last_question": data["chat_history"][-1][0] if data.get("chat_history") else "No messages"
                })
            except Exception as e:
                logger.error(f"Error reading session {filepath.name}: {e}")
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted, False if not found
        """
        filepath = self.storage_dir / f"{session_id}.json"
        
        if not filepath.exists():
            return False
        
        filepath.unlink()
        logger.info(f"Deleted chat session: {session_id}")
        
        if self.current_session and self.current_session.session_id == session_id:
            self.current_session = None
        
        return True
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current active session"""
        return self.current_session
    
    def add_to_current(self, question: str, answer: str):
        """
        Add Q&A to current session and save
        
        Args:
            question: User question
            answer: System answer
        """
        if not self.current_session:
            raise ValueError("No active session. Create or load a session first.")
        
        self.current_session.add_exchange(question, answer)
        self.save_session(self.current_session)
    
    def clear_current_history(self):
        """Clear history of current session"""
        if not self.current_session:
            raise ValueError("No active session.")
        
        self.current_session.clear_history()
        self.save_session(self.current_session)
        logger.info(f"Cleared history for session: {self.current_session.session_id}")
