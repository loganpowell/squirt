"""
Validation Transform Functions

Check output validity against schemas and constraints.
"""

import json
from typing import Any, Callable, Dict, List


def json_valid_transform(inputs: Dict[str, Any], output: Any) -> bool:
    """
    Check if output is valid JSON (for string outputs).

    Returns True for dict/list outputs, or for strings that parse as JSON.
    """
    if isinstance(output, (dict, list)):
        return True
    if isinstance(output, str):
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
    return False


def has_required_fields_transform(
    required_fields: List[str],
) -> Callable[[Dict[str, Any], Any], bool]:
    """
    Factory to check if output has required fields.

    Args:
        required_fields: List of field names that must be present.
            Supports nested fields with dot notation: "user.name", "data.items"

    Returns:
        Transform that returns True if all fields present

    Example:
        has_user = has_required_fields_transform(["user.id", "user.name"])

        @track(metrics=[
            Metric("has_required", has_user, AggregationType.AVERAGE)
        ])
        def my_component(text: str) -> dict:
            ...
    """

    def transform(inputs: Dict[str, Any], output: Any) -> bool:
        if not isinstance(output, dict):
            return False

        def check_nested(obj: Dict[str, Any], fields: List[str]) -> bool:
            for field in fields:
                parts = field.split(".")
                current: Any = obj
                for part in parts:
                    if not isinstance(current, dict) or part not in current:
                        return False
                    current = current[part]
            return True

        return check_nested(output, required_fields)

    return transform


def error_free_transform(inputs: Dict[str, Any], output: Any) -> bool:
    """
    Check if output contains no error indicators.

    Returns False if output contains common error keys:
    - "error", "errors", "exception", "traceback", "fault"
    """
    if not isinstance(output, dict):
        return True

    # Check for common error keys
    error_keys = {"error", "errors", "exception", "traceback", "fault"}
    if any(key in output for key in error_keys):
        return False

    return True


def create_pydantic_validation_transform(
    schema_class: type,
) -> Callable[[Dict[str, Any], Any], bool]:
    """
    Factory to create a Pydantic validation transform.

    Args:
        schema_class: Pydantic model class to validate against

    Returns:
        Transform function that returns True if valid, False otherwise

    Example:
        from pydantic import BaseModel

        class UserResponse(BaseModel):
            id: int
            name: str

        validate_user = create_pydantic_validation_transform(UserResponse)

        @track(metrics=[
            Metric("valid_response", validate_user, AggregationType.AVERAGE)
        ])
        def get_user(user_id: int) -> dict:
            ...
    """

    def transform(inputs: Dict[str, Any], output: Any) -> bool:
        try:
            if isinstance(output, dict):
                schema_class(**output)
            else:
                schema_class(data=output)
            return True
        except Exception:
            return False

    return transform


__all__ = [
    "json_valid_transform",
    "has_required_fields_transform",
    "error_free_transform",
    "create_pydantic_validation_transform",
]
