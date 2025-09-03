from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
import os
import openai
from download_tool import DownloadTool
from eval import run_batch_to_json
from framehub import FrameHub
from framehub_tools import build_tools
from picker_tool import PickerTool
from search_tool import SearchTool
from dotenv import load_dotenv

# load API-KEY
load_dotenv()
search_tool = SearchTool()
frame_hub = FrameHub()
download_tool = DownloadTool(frame_hub)
#model_str = 'google/gemini-2.5-flash'
model_str = 'openai/gpt-4.1'
#model_str = 'deepseek/deepseek-r1'
picker_tool = PickerTool(model=model_str)
framehub_tools = build_tools(frame_hub)
TOOLS = [search_tool, download_tool, *framehub_tools]
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    #api_key = os.getenv('OPENAI_API_KEY'),
    #model="anthropic/claude-sonnet-4",
    #model = 'google/gemini-2.5-flash',
    model = model_str,
    #model = 'meta-llama/llama-3.1-70b-instruct',
    temperature = 0
    # optional but recommended by OpenRouter:
)  # or your provider
llm_with_tools = llm.bind_tools(TOOLS)
#llm_with_tools = llm.bind_tools(TOOLS).bind(tool_choice="required")

#llm = ChatAnthropic(
#    api_key=os.getenv("OPENROUTER_API_KEY"),
#    base_url="https://openrouter.ai/api/v1",
    #api_key = os.getenv('OPENAI_API_KEY'),
#    model="anthropic/claude-sonnet-4",
    #temperature = 0
    # optional but recommended by OpenRouter:
#)  # or your provider
#llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an Open Data QA agent. You receive questions and should provide answers with the 
corresponding source. Formulate a search query und use the search tool. If you get too 
many (more than 30) or too little (zero) results, adapt the query. Govdata concatenates query terms via AND operator. 
Hence, multiple query terms reduce the answer set. Select the most promising dataset(s). 
In some cases, you do not need to have a look at the data, you simply need to provide the "dataset uri". 
For most questions, you need to have a look at the data, then select the 
downloadURL (when not given use the accessURL) for the most suitable dataset and download the data. Most of the time, 
you only need one dataset. If the download with the downloadURL fails, use the accessURL (in case given). Once the data 
is downloaded, have a look at it and answer the question. In your answer, provide the dataset uri as source.
Format you final answer in the field "FinalAnswer."
"""


def assistant_node(state: AgentState) -> AgentState:
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + state.get("messages",[])
    ai = llm_with_tools.invoke(msgs)
    #ai = llm.invoke(msgs)
    return {**state, "messages": state.get("messages",[]) + [ai]}


tool_node = ToolNode(TOOLS)


def summarize_node(state: AgentState) -> AgentState:
    msgs = [{"role": "system", "content": "Answer the input question with the data provided"}] + state["messages"]
    final = llm.invoke(msgs)
    return {**state, "messages": state["messages"] + [AIMessage(content=final.content)]}


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)
builder.add_node("summarize", summarize_node)
# Entry
builder.add_edge(START, "assistant")
# If the assistant requested a tool, go to tools; otherwise end with summarize
#builder.add_conditional_edges("assistant", tools_condition, {"tools":"tools", "end":"summarize"})
# After tools run, go back to assistant to continue the loop
#builder.add_edge("tools", "assistant")
# End
#builder.add_edge("summarize", END)


# ✅ Use END (or "__end__") here
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", END: "summarize"}   # or {"tools": "tools", "__end__": "summarize"}
)

builder.add_edge("tools", "assistant")
builder.add_edge("summarize", END)

# Optional checkpointer so multi-turn tool loops persist
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

#ids = [144, 157, 159, 166, 167, 176, 179, 183]
#for id in ids:
run_batch_to_json(model_str, builder,
                  "open-data-benchmark/de-questions.csv",
                  "results",
                  prefix="gpt4", recursion_limit=25, start_index=124)
# question = "Wie viele Walnussbäume gibt es bei der Aktion Gelbes Band - Naschbäume im Landkreis Regensburg?" # Optional checkpointer so multi-turn tool loops persist""
# memory = MemorySaver()
# initial_state: AgentState = {
#     "messages": [HumanMessage(content=question)],
# }
#
#
# result_state = graph.invoke(initial_state, config={
#     "recursion_limit": 50,
#     "configurable": {"thread_id": "run-40",}})
# print("FINAL ANSWER:\n", result_state["messages"][-1].content)