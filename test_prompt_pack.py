#!/usr/bin/env python3
"""Tests for :class:`PromptPack` prompt generation."""

from models.character import Character
from models.scenario import Scenario
from models.prompt_pack import PromptPack

def test_prompt_pack_generation():
    
    # Create test character
    character = Character(
        name="Test Character",
        age=25,
        gender="non-binary",
        role="companion",
        relationship="friend",
        mood={"optimism": 0.75, "patience": 0.45},
        personality={"empathy": 0.8, "humor": 0.6, "formality": 0.3},
        traits=["friendly", "curious"],
        backstory="A test character for debugging",
        voice_tone="casual"
    )
    
    # Create test scenario
    scenario = Scenario(
        setting="Test Environment",
        time_period="present day",
        objective="test the storage system",
        conflict="debugging issues",
        rules=["be thorough", "check for errors"]
    )
    
    # Create prompt pack
    pack = PromptPack(
        name="Test Pack",
        description="A test prompt pack",
        character=character,
        scenario=scenario
    )
    
    # Generate system prompt
    prompt = pack.generate_system_prompt()
    
    # Check if all expected elements are present
    expected_elements = [
        "Test Character",
        "25-year-old",
        "non-binary",
        "companion",
        "Relationship: friend",
        "Optimism: high",
        "Patience: moderate",
        "Test Environment",
        "present day",
        "very empathy",
        "moderately humor",
        "not very formality",
        "casual tone",
        "test the storage system",
        "A test character for debugging",
        "friendly, curious",
        "debugging issues",
        "be thorough",
        "check for errors"
    ]
    
    for element in expected_elements:
        assert element in prompt

