"""
Base state management with Observer pattern.

This module provides the foundation for reactive state management
throughout the EEG Viewer application.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Set
from copy import deepcopy


@dataclass
class BaseState:
    """
    Base class for all state objects with Observer pattern support.
    
    Features:
    - Subscribe/unsubscribe callbacks for state changes
    - Automatic notification when attributes change
    - Deep copy for state snapshots
    
    Usage:
        state = ViewerState()
        state.subscribe(lambda attr, old, new: print(f"{attr}: {old} -> {new}"))
        state.view_duration = 10.0  # Triggers notification
    """
    _observers: Set[Callable[[str, Any, Any], None]] = field(
        default_factory=set, repr=False, compare=False
    )
    _notify_enabled: bool = field(default=True, repr=False, compare=False)
    
    def subscribe(self, callback: Callable[[str, Any, Any], None]) -> Callable[[], None]:
        """
        Subscribe to state changes.
        
        Args:
            callback: Function called with (attr_name, old_value, new_value)
            
        Returns:
            Unsubscribe function
        """
        self._observers.add(callback)
        return lambda: self._observers.discard(callback)
    
    def unsubscribe(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Remove a callback from observers."""
        self._observers.discard(callback)
    
    def notify(self, attr: str, old_value: Any, new_value: Any) -> None:
        """Notify all observers of a state change."""
        if not self._notify_enabled:
            return
        for callback in list(self._observers):
            try:
                callback(attr, old_value, new_value)
            except Exception as e:
                print(f"[BaseState] Observer error for '{attr}': {e}")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override to notify observers on attribute changes."""
        # Skip internal attributes
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
            
        old_value = getattr(self, name, None)
        object.__setattr__(self, name, value)
        
        # Only notify if value actually changed
        if old_value != value:
            self.notify(name, old_value, value)
    
    def batch_update(self, **kwargs) -> None:
        """
        Update multiple attributes without triggering notifications for each.
        Triggers a single 'batch_update' notification at the end.
        """
        self._notify_enabled = False
        changes = {}
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    old_value = getattr(self, key)
                    if old_value != value:
                        changes[key] = (old_value, value)
                        setattr(self, key, value)
        finally:
            self._notify_enabled = True
        
        if changes:
            self.notify('batch_update', None, changes)
    
    def snapshot(self) -> dict:
        """Return a deep copy of the state as a dictionary."""
        return {
            k: deepcopy(v) 
            for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def reset(self) -> None:
        """Reset state to default values. Override in subclasses."""
        pass


class StateHolder:
    """
    Singleton-like container for state instances.
    
    Provides centralized access to all application states
    without using global variables.
    
    Usage:
        holder = StateHolder()
        holder.viewer = ViewerState()
        holder.pipeline = PipelineState()
        
        # Access from anywhere
        viewer_state = holder.viewer
    """
    _instance: StateHolder | None = None
    
    def __new__(cls) -> StateHolder:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._states = {}
        return cls._instance
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._states[name] = value
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        states = object.__getattribute__(self, '_states')
        if name in states:
            return states[name]
        raise AttributeError(f"State '{name}' not registered")
    
    def register(self, name: str, state: BaseState) -> None:
        """Register a state with a given name."""
        self._states[name] = state
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a state by name, returning default if not found."""
        return self._states.get(name, default)
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

