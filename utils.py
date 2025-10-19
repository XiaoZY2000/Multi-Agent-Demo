import re
import json
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from tqdm import tqdm
from IPython.display import display, Image

class ChatMessage(TypedDict):
    role: str
    receiver: list[str]
    content: str

class State(TypedDict):
    source_text: str
    compared_text_one: str
    compared_text_two: str
    chat_history: list[ChatMessage]
    agent_sequence: list[str]
    role_description: dict[str, str]
    turn: int

class MultiAgentEvaluator:
    def __init__(self, llm_agents, agent_sequence, role_description, prompt_template, final_prompt_to_use, pattern, max_turn=2):
        self.llm_agents = llm_agents
        self.agent_sequence = agent_sequence
        self.role_description = role_description
        self.max_turn = max_turn
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        self.final_prompt = final_prompt_to_use
        self.pattern = re.compile(pattern)
        self.graph = self._create_graph()

    def _agent_fn_factory(self, agent_idx):
        def agent_fn(state: State):
            current_agent = state["agent_sequence"][state["turn"] % len(self.llm_agents)]
            agent_history = [
                msg["content"] for msg in state["chat_history"]
                if current_agent in msg["receiver"] or msg["receiver"] == ["all"]
            ]
            final_prompt = self.final_prompt if (state["turn"] // len(self.llm_agents)) + 1 == self.max_turn else ""
            prompt = self.prompt_template.format_messages(
                source_text=state["source_text"],
                compared_text_one=state["compared_text_one"],
                compared_text_two=state["compared_text_two"],
                chat_history=agent_history,
                role_description=self.role_description[current_agent],
                agent_name=current_agent,
                final_prompt=final_prompt
            )
            prompt_text = "\n".join([msg.content for msg in prompt])
            response = self.llm_agents[agent_idx].invoke(prompt_text).content
            chat_message = ChatMessage(role=current_agent, receiver=["all"], content=response)
            return {"chat_history": state["chat_history"] + [chat_message], "turn": state["turn"] + 1}
        return agent_fn

    def _create_graph(self):
        builder = StateGraph(State)
        for i in range(len(self.llm_agents)):
            builder.add_node(f"agent{i}", self._agent_fn_factory(i))
        builder.add_edge(START, "agent0")
        for i in range(len(self.llm_agents) - 1):
            builder.add_edge(f"agent{i}", f"agent{i+1}")
        builder.add_conditional_edges(f"agent{len(self.llm_agents) - 1}", self._should_continue,
                                      {"continue": "agent0", "end": END})
        return builder.compile()

    def _should_continue(self, state):
        return "continue" if (state["turn"] // len(self.llm_agents)) < self.max_turn else "end"

    def evaluate(self, question_obj):
        initial_state = {
            "source_text": question_obj["question"],
            "compared_text_one": question_obj["response"]["gpt35"],
            "compared_text_two": question_obj["response"]["vicuna"],
            "chat_history": [],
            "agent_sequence": self.agent_sequence,
            "role_description": self.role_description,
            "turn": 0
        }
        final_state = self.graph.invoke(initial_state)
        print("Final turn: ", final_state["turn"])
        results = []
        for i, agent_name in tqdm(enumerate(self.agent_sequence), desc="Evaluating Agents", total=len(self.agent_sequence)):
            agent_reply = final_state["chat_history"][-len(self.agent_sequence) + i]["content"]
            match = re.search(self.pattern, agent_reply)
            if match:
                score_1 = int(match.group(1))
                score_2 = int(match.group(2))
                results.append({
                    "role": agent_name,
                    "evaluation": agent_reply,
                    "score": [score_1, score_2]
                })
                print(f"Agent {i}'s response recorded!")
            else:
                print(f"Error: Agent {i}'s response does not match the expected pattern.")
                results.append({
                    "role": agent_name,
                    "evaluation": agent_reply,
                    "score": [0, 0]  # Default score if pattern does not match
                })
        return {
            "question": question_obj["question"],
            "response": question_obj["response"],
            "evaluation": results
        }

    def evaluate_batch(self, data_list):
        outputs = []
        for i, item in enumerate(data_list):
            print(f"Evaluating item {i + 1}/{len(data_list)}...")
            result = self.evaluate(item)
            outputs.append(result)
        return outputs

    def save_results(self, results, path="output.json"):
        with open(path, "w") as f:
            json.dump(results, f, indent=4)

    def show_graph(self):
        graph_image = self.graph.get_graph().draw_mermaid_png()
        with open("MAS_structure.png", "wb") as f:
            f.write(graph_image)
