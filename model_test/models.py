from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class InitialCartItem:
    product_name: str
    quantity: int


@dataclass
class InitialCartState:
    items: list[InitialCartItem]


@dataclass
class ExpectedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedToolPath:
    name: str
    tools: list[ExpectedToolCall]
    description: str = ""


@dataclass
class TestCase:
    name: str
    prompt: str
    expected_tools_variants: list[ExpectedToolPath]
    initial_cart_state: InitialCartState | None = None


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class AgentResponse:
    tool_calls: list[ToolCall]
    llm_requests: int
    llm_total_time: float
    final_message: str = ""


@dataclass
class AgentTestResult:
    test_case: TestCase
    success: bool
    response_time: float
    response: AgentResponse | None = None
    matched_path: str = ""
    error_message: str = ""


@dataclass
class AgentReport:
    timestamp: datetime
    results: list[AgentTestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_llm_time: float
    avg_time_per_req: float
