from utils import MultiAgentEvaluator
from argparse import ArgumentParser
import yaml
import json
from langchain_openai import ChatOpenAI

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

with open("test.json", "r") as file:
    data = json.load(file)

llm_1 = ChatOpenAI(
    model="phi4-mini-reasoning",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)

llm_2 = ChatOpenAI(
    model="phi4-mini-reasoning",
    openai_api_base="http://localhost:11435/v1",
    openai_api_key="ollama",
)

# LLM numbers should match the number of agents
llms = [llm_1, llm_2]

# Agents can be chosen from ['General Public', 'Critic', 'News Author', 'Psychologist', 'Scientist']
agents = ['General Public', 'Critic']

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    prompt_template = config["prompt_template"]
    final_prompt_to_use = config["final_prompt"]
    pattern = config["pattern"]
    role_description = config["role_description"]

    MAS = MultiAgentEvaluator(
        llm_agents=llms,
        agent_sequence=agents,
        role_description=role_description,
        prompt_template=prompt_template,
        final_prompt_to_use=final_prompt_to_use,
        pattern=pattern,
        max_turn=2
    )
    print("Multi-Agent System Created!")

    MAS.show_graph()
    # print(MAS.pattern)
    responses = MAS.evaluate_batch(data)

    print("Saving Responses...")
    MAS.save_results(responses, "output.json")
    print("Done saving responses!")


if __name__ == "__main__":
    main()