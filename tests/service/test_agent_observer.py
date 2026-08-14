"""验证 Agent 观测回调的统计、日志与 Runtime 挂载行为。"""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage

from src.service.agent.chat_observer import AgentObservationHandler
from src.service.runtime import Runtime


def test_handler_aggregates_metrics_and_logs_summary(caplog) -> None:
    """回调应累计模型/工具/检索调用，并输出统一格式的汇总日志。"""

    handler = AgentObservationHandler(
        thread_id="thread-1",
        question="测试问题",
    )
    model_run_id = uuid.uuid4()
    tool_run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"name": "ChatOpenAI"},
        [[HumanMessage(content="你好")]],
        run_id=model_run_id,
    )
    handler.on_llm_end(
        SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata={
                                "input_tokens": 1,
                                "output_tokens": 2,
                                "total_tokens": 3,
                            }
                        )
                    )
                ]
            ]
        ),
        run_id=model_run_id,
    )
    handler.on_tool_start(
        {"name": "retrieve_knowledge_files"},
        "查询",
        run_id=tool_run_id,
    )
    handler.on_tool_end("结果", run_id=tool_run_id)
    handler.on_retriever_start({}, "查询")
    handler.on_retriever_end([1, 2])
    handler.on_agent_action(SimpleNamespace(tool="retrieve_knowledge_files"))

    with caplog.at_level(
        logging.INFO,
        logger="src.service.agent.chat_observer",
    ):
        handler.summary()

    assert "threadId=thread-1" in caplog.text
    assert "modelCalls=1" in caplog.text
    assert "totalTokens=3" in caplog.text
    assert "toolCalls=1" in caplog.text
    assert "retrievalCalls=1" in caplog.text
    assert "agentSteps=1" in caplog.text
    assert "errorCount=0" in caplog.text


def test_debug_observation_logs_never_include_prompt_or_tool_text(caplog) -> None:
    secret = "PRIVATE-CREDENTIAL-CONTENT"
    handler = AgentObservationHandler(thread_id="thread-1", question=secret)
    model_run_id = uuid.uuid4()
    tool_run_id = uuid.uuid4()

    with caplog.at_level(logging.DEBUG, logger="src.service.agent.chat_observer"):
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content=secret)]],
            run_id=model_run_id,
        )
        handler.on_tool_start(
            {"name": "retrieve_knowledge_files"},
            secret,
            run_id=tool_run_id,
        )
        handler.on_retriever_start({}, secret)
        handler.on_llm_error(RuntimeError(secret), run_id=model_run_id)
        handler.on_tool_error(RuntimeError(secret), run_id=tool_run_id)

    assert secret not in caplog.text
    assert "inputLength=" in caplog.text
    assert "queryLength=" in caplog.text
    assert "Sha256=" not in caplog.text
    assert "exceptionType=RuntimeError" in caplog.text


def test_state_trace_never_outputs_or_hashes_state_content(capsys) -> None:
    secret = "LOW-ENTROPY-SECRET"

    Runtime._print_state("result", {"messages": [secret]})

    output = capsys.readouterr().out
    assert secret not in output
    assert "sha256" not in output.casefold()
    assert "type=dict" in output
    assert "itemCount=1" in output


def test_answer_attaches_observation_handler_when_enabled() -> None:
    """启用观测时，agent.invoke 的 config 应携带回调实例。"""

    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = Runtime.__new__(Runtime)
    runtime.retrieval_context = object()
    runtime.agent = runnable
    runtime._trace_state = False
    runtime._observation_enabled = True
    runtime._result_text = lambda result: "回答"

    answer, title = runtime._answer("问题", thread_id="thread-1")

    assert answer == "回答"
    assert title is None
    config = runnable.invoke.call_args.kwargs["config"]
    assert "callbacks" in config
    handler = config["callbacks"][0]
    assert isinstance(handler, AgentObservationHandler)
    assert handler.thread_id == "thread-1"


def test_answer_skips_observation_when_disabled() -> None:
    """关闭观测时，config 不应包含 callbacks。"""

    runnable = Mock()
    runnable.invoke.return_value = {"messages": [AIMessage(content="回答")]}
    runtime = Runtime.__new__(Runtime)
    runtime.retrieval_context = object()
    runtime.agent = runnable
    runtime._trace_state = False
    runtime._observation_enabled = False
    runtime._result_text = lambda result: "回答"

    runtime._answer("问题", thread_id="thread-1")

    config = runnable.invoke.call_args.kwargs["config"]
    assert "callbacks" not in config


def test_thread_config_accepts_callbacks() -> None:
    """_thread_config 应在传回调时附加 callbacks 键。"""

    handler = AgentObservationHandler(thread_id="thread-1")
    config = Runtime._thread_config("thread-1", callbacks=[handler])

    assert config["configurable"] == {"thread_id": "thread-1"}
    assert config["recursion_limit"] == 25
    assert config["callbacks"] == [handler]
