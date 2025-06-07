import pytest
import os
import tempfile
import shutil
import torch
from unittest.mock import MagicMock, patch
from typing import List, Optional

from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn
from hive.ml.data.cache import SQLiteCache
from hive.game_engine.game_state import Game
from torch_geometric.data import Data

@pytest.fixture
def cache():
    temp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(temp_dir, "test_cache.db")
    yield SQLiteCache(cache_path)
    shutil.rmtree(temp_dir)

def test_cache_set_get(cache):
    """Test basic set and get operations"""
    test_data = {"key1": "value1", "key2": [1, 2, 3]}
    cache.set("test_key", test_data)
    retrieved = cache.get("test_key")
    assert test_data == retrieved

def test_cache_get_nonexistent(cache):
    """Test getting a non-existent key returns None"""
    assert cache.get("nonexistent_key") is None

def test_cache_clear(cache):
    """Test clearing the cache"""
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert cache.get_size() == 2
    cache.clear()
    assert cache.get_size() == 0

def test_cache_get_size(cache):
    """Test getting cache size"""
    assert cache.get_size() == 0
    cache.set("key1", "value1")
    assert cache.get_size() == 1
    cache.set("key2", "value2")
    assert cache.get_size() == 2

@pytest.fixture
def mock_loader():
    mock = MagicMock()
    game_mock = MagicMock(spec=Game)
    game_mock.player_turns = {'WHITE': 0, 'BLACK': 0}  # Use integers instead of lists
    game_mock.queens = {'WHITE': None, 'BLACK': None}
    game_mock.parent = None
    game_mock.move = None  # Add move attribute to fix AttributeError
    game_mock.current_turn = 'WHITE'  # Add current_turn attribute
    game_mock.grid = {}  # Add empty grid
    game_mock.unplayed_pieces = {'WHITE': [], 'BLACK': []}  # Add empty unplayed pieces
    mock.get_game.return_value = game_mock
    mock.__len__.return_value = 10
    return mock

@pytest.fixture
def mock_cache():
    mock = MagicMock()
    mock.get.return_value = None
    return mock

@pytest.fixture
def dataset(mock_loader, mock_cache):
    with patch('hive.ml.data.dataset.GameDataLoader', return_value=mock_loader):
        with patch('hive.ml.data.dataset.SQLiteCache', return_value=mock_cache):
            return HiveLazyGameDataset(
                filepath="test_path.txt",
                batch_size=10,
                use_cache=True
            )

def test_dataset_init(dataset, mock_loader, mock_cache):
    """Test dataset initialization"""
    assert dataset.filepath == "test_path.txt"
    assert dataset.batch_size == 10
    assert dataset.use_cache is True
    assert dataset.cache is mock_cache

def test_get_with_cache_hit(dataset, mock_cache):
    """Test get() with cache hit"""
    mock_cache.get.return_value = ["cached_data"]
    result = dataset.get(0)
    assert result == ["cached_data"]
    dataset.loader.get_game.assert_not_called()

def test_get_with_cache_miss(dataset, mock_cache):
    """Test get() with cache miss"""
    with patch('hive.ml.data.dataset.process_endgame') as mock_process:
        mock_process.return_value = ["processed_data"]
        result = dataset.get(0)
        assert result == ["processed_data"]
        mock_cache.set.assert_called_once_with(f"{dataset.filepath}:0", ["processed_data"])

def test_is_cached(dataset, mock_cache):
    """Test is_cached() method"""
    dataset.is_cached(0)
    mock_cache.get.assert_called_once_with("test_path.txt:0")

def test_clear_cache(dataset, mock_cache):
    """Test clear_cache() method"""
    dataset.clear_cache()
    mock_cache.clear.assert_called_once()

def test_get_invalid_index(dataset, mock_loader):
    """Test get() with invalid index"""
    mock_loader.get_game.return_value = None
    result = dataset.get(999)
    assert result is None

def test_collate_fn():
    """Test the collate function with valid data"""
    data1 = Data(x=torch.tensor([1, 2]), edge_index=torch.tensor([[0], [1]]))
    data2 = Data(x=torch.tensor([3, 4]), edge_index=torch.tensor([[1], [0]]))
    batch = [[data1], [data2]]
    result = collate_fn(batch)
    assert result is not None
    assert result.x.shape[0] == 4  # 2 nodes per graph, 2 graphs = 4 total nodes
    assert result.batch.tolist() == [0, 0, 1, 1]  # Batch indices

def test_collate_fn_empty():
    """Test collate function with empty batch"""
    assert collate_fn([]) is None
    assert collate_fn([None, None]) is None