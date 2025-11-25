import os
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client
#load_dotenv()


# run the agent for a specified question (identified by its id)
# reads from the input file (usually {de,en}-questions.csv from the open-data-benchmark folder)
def run_agent(model, builder, input_file, results_dir, *, prefix, recursion_limit=50, start_question_id=1, limit=None):
    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    memory = MemorySaver()
    # build the graph
    graph = builder.compile(checkpointer=memory)
    json_files = []

    # Open input CSV with all the questions
    with open(input_file, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        # when the agent is evaluated, the starting_question_id is provided
        # to reach the corresponding row with the help of the reader, we need the index,
        # which is start_question_id minus 1
        for _ in range(start_question_id - 1):
            next(reader, None)

        for i, row in enumerate(reader, start=start_question_id):
            if limit is not None and (i - start_question_id + 1) > limit:
                break

            # from the input row metadata is saved to make evaluation later easier
            thread_id = f"{prefix}-{i:04d}"
            question_id = i
            question = row.get("frage", "") or ""
            answer = row.get("antwort", "") or ""
            question_type = row.get("frage_typ", "") or ""
            source = row.get("datengrundlage", "") or ""
            remark = row.get("bemerkungen", "") or ""
            # the agent gets the question, when langfuse is turned on,
            # worflow executions can be tracked
            try:
                initial_state = {"messages": [HumanMessage(content=question)]}
                Langfuse()
                langfuse_handler = CallbackHandler()

                # execute the agentic workflow
                result_state = graph.invoke(
                    initial_state,
                    config={
                        "recursion_limit": recursion_limit,
                        "configurable": {"thread_id": thread_id},
                        "callbacks": [langfuse_handler],
                        "metadata": {
                            "langfuse_user_id": "random-user",
                            "langfuse_session_id": "random-session",
                            "langfuse_tags": [model,thread_id,source,question_type]
                        }
                    },
                )

                # read the last message
                messages = result_state["messages"]
                llm_text = ""
                if messages:  # make sure the list is not empty
                    last_message = messages[-1]
                    llm_text = getattr(last_message, "content", "")
                    # in case the last message is empty, read the second last message
                    if llm_text == "":
                        if len(messages) > 1:
                            second_last_msg = messages[-2]
                            llm_text = getattr(second_last_msg, "content", "") or ""

                # OpenAI, Llama
                #final_msg = result_state["messages"][-1]
                # Anthropic, Google, Deepseek
                #final_msg = result_state["messages"][-2]
                #print(result_state["messages"])
                #llm_text = getattr(final_msg, "content", "") or ""

            # catch any error message for error type counting
            except Exception as e:
                llm_text = f"ERROR: {str(e)}"

            # Build JSON object for this row
            data = {
                "thread_id": thread_id,
                "run_at": datetime.utcnow().isoformat() + "Z",
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "source": source,
                "remark": remark,
                "llm_final": llm_text
            }

            # Write the result with the llm answer to a json file
            out_path = os.path.join(results_dir, f"{thread_id}.json")
            with open(out_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=2)

            json_files.append(out_path)
            print(f"Wrote {out_path}")

    print(f"Done. Wrote {len(json_files)} JSON files to {results_dir}.")
    return json_files
