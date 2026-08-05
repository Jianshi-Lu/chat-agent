"""真实 MySQL 独立库上的 LangGraph Checkpointer 持久化验收。

需要环境变量 ``CHAT_AGENT_CHECKPOINT_DATABASE_URL`` 指向独立的 MySQL 8.x 数据库，
例如 ``mysql://chat_agent_checkpoint:xxx@127.0.0.1:3306/chat_agent_checkpoint``。
未配置时整组跳过；本文件只操作 Checkpointer 表，不触碰 ``chat_agent`` 业务表。
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pymysql
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver


DATABASE_URL = os.environ.get("CHAT_AGENT_CHECKPOINT_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="需要 CHAT_AGENT_CHECKPOINT_DATABASE_URL 指向独立 MySQL 库",
)


def _make_checkpointer() -> tuple[PyMySQLSaver, object]:
    """创建持有长连接的最小 Saver，避免 with 块退出时提前释放连接。"""

    conn = pymysql.connect(
        **PyMySQLSaver.parse_conn_string(DATABASE_URL),
        autocommit=True,
    )
    saver = PyMySQLSaver(conn)
    saver.setup()
    return saver, conn


def _message_texts(state: object) -> list[str]:
    """提取 checkpoint State 中全部消息文本。"""

    return [
        getattr(message, "content", "")
        for message in state.values["messages"]
    ]


def test_same_thread_second_round_reads_first_round_state() -> None:
    """相同 thread_id 连续两轮，第二轮必须读到第一轮的 State。"""

    thread_id = f"same-thread-{uuid.uuid4()}"
    saver, conn = _make_checkpointer()
    try:
        agent = create_agent(
            model=FakeMessagesListChatModel(
                responses=[
                    AIMessage(content="回答一"),
                    AIMessage(content="回答二"),
                ]
            ),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": thread_id}}
        agent.invoke(
            {"messages": [HumanMessage(content="问题一")]},
            config=config,
        )
        agent.invoke(
            {"messages": [HumanMessage(content="问题二")]},
            config=config,
        )

        contents = _message_texts(agent.get_state(config))
        assert "问题一" in contents
        assert "回答一" in contents
        assert "问题二" in contents
        assert "回答二" in contents
    finally:
        saver.delete_thread(thread_id)
        conn.close()


def test_restart_process_keeps_state_for_same_thread() -> None:
    """重启 Python 进程后，相同 thread_id 仍能恢复此前写入的 State。"""

    thread_id = f"restart-{uuid.uuid4()}"
    script = f"""
import os
import pymysql
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

url = os.environ["CHAT_AGENT_CHECKPOINT_DATABASE_URL"]
conn = pymysql.connect(**PyMySQLSaver.parse_conn_string(url), autocommit=True)
saver = PyMySQLSaver(conn)
saver.setup()
agent = create_agent(
    model=FakeMessagesListChatModel(responses=[AIMessage(content="进程一回答")]),
    checkpointer=saver,
)
agent.invoke(
    {{"messages": [HumanMessage(content="进程一问题")]}},
    config={{"configurable": {{"thread_id": {thread_id!r}}}}},
)
conn.close()
print("subprocess-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "CHAT_AGENT_CHECKPOINT_DATABASE_URL": DATABASE_URL,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "subprocess-ok" in result.stdout

    saver, conn = _make_checkpointer()
    try:
        agent = create_agent(
            model=FakeMessagesListChatModel(
                responses=[AIMessage(content="进程二回答")]
            ),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": thread_id}}
        agent.invoke(
            {"messages": [HumanMessage(content="进程二问题")]},
            config=config,
        )

        contents = _message_texts(agent.get_state(config))
        assert "进程一问题" in contents
        assert "进程一回答" in contents
        assert "进程二问题" in contents
        assert "进程二回答" in contents
    finally:
        saver.delete_thread(thread_id)
        conn.close()


def test_other_thread_state_is_fully_isolated() -> None:
    """不同 thread_id 的 State 必须完全隔离。"""

    thread_a = f"thread-a-{uuid.uuid4()}"
    thread_b = f"thread-b-{uuid.uuid4()}"
    saver, conn = _make_checkpointer()
    try:
        agent = create_agent(
            model=FakeMessagesListChatModel(
                responses=[
                    AIMessage(content="A 回答一"),
                    AIMessage(content="A 回答二"),
                    AIMessage(content="B 回答"),
                ]
            ),
            checkpointer=saver,
        )
        config_a = {"configurable": {"thread_id": thread_a}}
        config_b = {"configurable": {"thread_id": thread_b}}
        agent.invoke(
            {"messages": [HumanMessage(content="A 问题一")]},
            config=config_a,
        )
        agent.invoke(
            {"messages": [HumanMessage(content="A 问题二")]},
            config=config_a,
        )
        agent.invoke(
            {"messages": [HumanMessage(content="B 问题")]},
            config=config_b,
        )

        contents_a = _message_texts(agent.get_state(config_a))
        contents_b = _message_texts(agent.get_state(config_b))
        assert "A 问题一" in contents_a
        assert "A 问题二" in contents_a
        assert "B 问题" not in contents_a
        assert contents_b == ["B 问题", "B 回答"]
    finally:
        saver.delete_thread(thread_a)
        saver.delete_thread(thread_b)
        conn.close()
