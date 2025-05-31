import sqlite3
import pickle
import os
from typing import Any, Optional
from pathlib import Path


class SQLiteCache:
    """
    A persistent cache using SQLite and pickle for storing processed game data.
    
    This cache stores serialized Python objects in an SQLite database,
    providing persistent storage between runs without consuming memory
    when not in use.
    """
    
    def __init__(self, cache_path: str):
        """
        Initialize the SQLite cache.
        
        Args:
            cache_path: Path to the SQLite database file
        """
        self.cache_path = cache_path
        self._init_db()
        
    def _init_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value BLOB
        )
        ''')
        conn.commit()
        conn.close()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
        
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        pickled_value = pickle.dumps(value)
        cursor.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, pickled_value))
        conn.commit()
        conn.close()
        
    def clear(self) -> None:
        """Clear all entries from the cache."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        
    def get_size(self) -> int:
        """
        Get the number of entries in the cache.
        
        Returns:
            The number of cached entries
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0

