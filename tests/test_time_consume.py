import asyncio
import importlib


def test_time_consume_measures_the_awaited_duration(monkeypatch):
    time_consume_module = importlib.import_module("decorator.time_consume")
    messages = []
    operation_started = False

    class FakeLogger:
        def info(self, message):
            messages.append(message)

    def fake_time():
        return 10.25 if operation_started else 10.0

    monkeypatch.setattr(time_consume_module, "logger", FakeLogger())
    monkeypatch.setattr(time_consume_module.time, "time", fake_time)

    @time_consume_module.time_consume
    async def operation():
        nonlocal operation_started
        await asyncio.sleep(0)
        operation_started = True
        return "完成"

    assert asyncio.run(operation()) == "完成"
    assert messages == ["operation 执行耗时: 0.2500 秒"]
