import pytest
from safety.safety_checker import SafetyChecker

def test_empty_query():
    is_valid, msg = SafetyChecker.is_valid_query("")
    assert not is_valid
    assert "empty" in msg.lower()

def test_short_query():
    is_valid, msg = SafetyChecker.is_valid_query("Hi")
    assert not is_valid
    assert "too short" in msg.lower()

def test_valid_query():
    is_valid, msg = SafetyChecker.is_valid_query("I have a fever and cough for two days.")
    assert is_valid
    assert msg == ""

def test_emergency_query():
    is_emergency, msg = SafetyChecker.check_emergency("I am experiencing severe chest pain and can't breathe")
    assert is_emergency
    assert "severe chest pain" in msg.lower()
    
def test_non_emergency_query():
    is_emergency, msg = SafetyChecker.check_emergency("I have a mild headache")
    assert not is_emergency
    assert msg == ""
