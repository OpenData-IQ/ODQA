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
from eval import run_agent
from table_register import TableRegister
from table_tools import build_tools
from search_tool import SearchTool
from dotenv import load_dotenv

# load API-KEY
load_dotenv()
search_tool = SearchTool()
frame_hub = TableRegister()
download_tool = DownloadTool(frame_hub)
#model_str = 'google/gemini-2.5-flash'
#model_str = 'mistralai/mistral-medium-3.1'
model_str = 'openai/gpt-5-mini'
#model_str= 'anthropic/claude-3.7-sonnet'
#model_str= 'deepseek/deepseek-r1'
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

SYSTEM_PROMPT = """You are an Open Data QA agent for Germany. You receive questions and should provide answers with the 
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
# assistant node decides whether to call tools
# or go to summarizer node when no tool calling is needed
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", END: "summarize"}
)

builder.add_edge("tools", "assistant")
builder.add_edge("summarize", END)

# Optional checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

ids = [5, 24, 25, 34, 35, 41, 44, 58, 70, 71, 74, 77, 80, 85,86,102,103,104,105,106,117,122,123,124,125,146,155,157,158,187,189,191,194,195,196,202]
for id in ids:
    run_agent(model_str, builder,
                  "open-data-benchmark/de-questions.csv",
                  "test",
                   prefix="gpt5-mini-2", recursion_limit=40, question_id=id)
