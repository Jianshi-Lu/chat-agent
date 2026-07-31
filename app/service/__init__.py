"""对话服务的外部接口包。

上层统一从 ``app.service.client`` 导入 ``ChatClient``。模型
Agent 和配置分别位于 ``app.agent`` 与根路径的 ``configuration`` 包。
本包不负责查询意图判断、检索器选择或上下文组装。
"""
