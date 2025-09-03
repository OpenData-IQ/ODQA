from typing import Optional, Type
from pydantic import BaseModel, Field, Extra
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os

class PickerToolInput(BaseModel):
    input_question: Optional[str] = Field(
        None, description="The initial question of the user"
    )
    search_output: Optional[str] = Field(
        None, description="The result table from querying govdata"
    )

class PickerTool(BaseTool):
    name: str = "select_dataset"
    description: str = (
        "Select the dataset most suitable for the input query."
    )
    args_schema: Type[BaseModel] = PickerToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=self.model,
        )

    def _run(self, input_question: Optional[str] = None, search_output: Optional[str] = None):
        # Guardrails (optional)
        if not input_question:
            return "ERROR: input_question was not provided to the tool."

        # Build messages with BOTH inputs
        msgs = [
            SystemMessage(
                content=(
                    "You are a dataset picker. "
                    "Given a user question and a list/table of candidate datasets, "
                    "return the single best **dataset URL**. "
                    "Output ONLY the URL (no extra text)."
                )
            ),
            HumanMessage(
                content=(
                    f"User question:\n{input_question}\n\n"
                    f"Candidate datasets / search results:\n{search_output or '(none)'}"
                )
            ),
        ]

        resp = self.llm.invoke(msgs)
        # Return plain text for the tool output
        return (getattr(resp, "content", None) or str(resp)).strip()
