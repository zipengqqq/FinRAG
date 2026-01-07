import uuid
from typing import List, Dict
from starlette.responses import StreamingResponse
from utils.logger_util import logger
from rag_graph import app as graph_workflow
from request.ask_request import AskRequest

class ChatService:
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def _get_messages(self, cid: str) -> List[Dict[str, str]]:
        return self._store.get(cid, [])

    def _append(self, cid: str, role: str, content: str):
        arr = self._store.get(cid)
        if arr is None:
            arr = []
            self._store[cid] = arr
        arr.append({"role": role, "content": content})

    def _history_str(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages[-10:]:
            parts.append(f"{m['role']}:{m['content']}")
        return "\n".join(parts)

    async def sse_response(self, req: AskRequest):
        cid = req.conversation_id or str(uuid.uuid4())
        messages = self._get_messages(cid)
        history_str = self._history_str(messages)
        user_content = req.query

        async def event_generator():
            collected = []
            try:
                state = {"query": req.query, "year": '', "history_str": history_str}
                async for event in graph_workflow.astream_events(state, version="v2"):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            text = chunk.content
                            collected.append(text)
                            yield f"data: {text}\n\n"
                logger.info(f"问题：{req.query}\nLLM响应内容：{''.join(collected)}")
                yield "data: [DONE]\n\n"
            finally:
                self._append(cid, "user", user_content)
                self._append(cid, "assistant", "".join(collected))

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )