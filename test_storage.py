#!/usr/bin/env python3
"""Tests for :class:`Storage` file operations."""

from pathlib import Path

from utils.storage import Storage
from models.character import Character
from models.scenario import Scenario

def test_character_storage(tmp_path):
    storage = Storage(base_path=tmp_path)
    
    # Create a test character
    character = Character(
        name="Test Character",
        age=25,
        gender="non-binary",
        role="companion",
        personality={"empathy": 0.8, "humor": 0.6, "formality": 0.3},
        traits=["friendly", "curious"],
        backstory="A test character for debugging",
        voice_tone="casual"
    )
    
    filepath = storage.save_character(character)
    assert Path(filepath).exists()

    loaded_character = storage.load_character(Path(filepath).name)
    assert loaded_character is not None
    assert loaded_character.model_dump() == character.model_dump()

    character_names = storage.get_character_names()
    assert Path(filepath).stem in character_names

def test_scenario_storage(tmp_path):
    storage = Storage(base_path=tmp_path)
    
    # Create a test scenario
    scenario = Scenario(
        setting="Test Environment",
        time_period="present day",
        objective="test the storage system",
        conflict="debugging issues",
        rules=["be thorough", "check for errors"]
    )
    
    filepath = storage.save_scenario(scenario)
    assert Path(filepath).exists()

    loaded_scenario = storage.load_scenario(Path(filepath).name)
    assert loaded_scenario is not None
    assert loaded_scenario.model_dump() == scenario.model_dump()

    scenario_names = storage.get_scenario_names()
    assert Path(filepath).stem in scenario_names

