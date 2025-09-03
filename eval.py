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


load_dotenv()

def run_batch_to_json(model, builder, input_file, results_dir, *, prefix="qa", recursion_limit=50, start_index=1, limit=None):
    """
    Reads question/answer/remark rows from input_file (CSV),
    runs them through the agent, and writes EACH ROW as its own JSON file
    into results_dir, named like: {prefix}-0001.json, {prefix}-0002.json, ...
    """

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    files_written = []

    # Open input CSV
    with open(input_file, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        # Skip rows up to start_index - 1
        for _ in range(start_index - 1):
            next(reader, None)

        for i, row in enumerate(reader, start=start_index):
            if limit is not None and (i - start_index + 1) > limit:
                break


            thread_id = f"{prefix}-{i:04d}"
            question_id = i
            question = row.get("frage", "") or ""
            answer = row.get("antwort", "") or ""
            question_type = row.get("frage_typ", "") or ""
            source = row.get("datengrundlage", "") or ""
            remark = row.get("bemerkungen", "") or ""

            try:
                initial_state = {"messages": [HumanMessage(content=question)]}
                #initial_state = {"messages": create_messages_with_synthesis_prompt(question, system_prompt) }
                # baseline (how many steps existed before this run)
                #baseline_steps = sum(1 for _ in graph.get_state_history(
                #    config={"configurable": {"thread_id": thread_id}}
                #))

                Langfuse()  # picks up env vars
                langfuse_handler = CallbackHandler()  # <-

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

                messages = result_state["messages"]

                # Default to empty string
                llm_text = ""

                if messages:  # make sure the list is not empty
                    final_msg = messages[-1]
                    llm_text = getattr(final_msg, "content", "")
                    if llm_text == "":
                        # fallback
                        if len(messages) > 1:
                            second_last_msg = messages[-2]
                            llm_text = getattr(second_last_msg, "content", "") or ""

                # OpenAI, Llama
                #final_msg = result_state["messages"][-1]
                # Anthropic, Google, Deepseek
                #final_msg = result_state["messages"][-2]
                #print(result_state["messages"])
                #llm_text = getattr(final_msg, "content", "") or ""

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

            # Write each row as its own JSON file, e.g., results/qa-0001.json
            out_path = os.path.join(results_dir, f"{thread_id}.json")
            with open(out_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=2)

            files_written.append(out_path)
            print(f"Wrote {out_path}")

    print(f"Done. Wrote {len(files_written)} JSON files to {results_dir}.")
    return files_written
