# Setup
# First, we need to install the packages required.
# !pip install --quiet -U langchain langchain_openai tavily-python
# key of GPT APT: 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
# Tavily API KEY: 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
# LangSmith KEY: ls__51705204d7ab4d10a06aa91d6cc8380d
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")
os.environ["LANGCHAIN_PROJECT"] = "Plan-and-execute"

# Define Tools
# We will first define the tools we want to use. 
# For this simple example, we will use a built-in search tool via Tavily. 
# However, it is really easy to create your own tools - see documentation here on how to do that.

from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]

# Define our Execution Agent
# Now we will create the execution agent we want to use to execute tasks. 
# Note that for this example, we will be using the same execution agent for each task, but this doesn't HAVE to be the case.

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4-turbo-preview")
# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
from langgraph.prebuilt import create_agent_executor
agent_executor = create_agent_executor(agent_runnable, tools)
agent_executor.invoke(
    {"input": "who is the winnner of the us open", "chat_history": []}
)

# Define the State
# Let's now start by defining the state the track for this agent.
# First, we will need to track the current plan. Let's represent that as a list of strings.
# Next, we should track previously executed steps. 
# Let's represent that as a list of tuples (these tuples will contain the step and then the result)
# Finally, we need to have some state to represent the final response as well as the original input.

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
# Planning Step
# Let's now think about creating the planning step. This will use function calling to create a plan.
    
from langchain_core.pydantic_v1 import BaseModel
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_template(
"""For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
{objective}""")
planner = create_structured_output_runnable(
    Plan, ChatOpenAI(model="gpt-4-turbo-preview", temperature=0), planner_prompt
)
planner.invoke(
    {"objective": "what is the hometown of the current Australia open winner?"}
)

# Re-Plan Step
# Now, let's create a step that re-does the plan based on the result of the previous step.

from langchain.chains.openai_functions import create_openai_fn_runnable


class Response(BaseModel):
    """Response to user."""

    response: str


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = create_openai_fn_runnable(
    [Plan, Response],
    ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
    replanner_prompt,
)

# Create the Graph
# We can now create the graph!

async def execute_step(state: PlanExecute):
    task = state["plan"][0]
    agent_response = await agent_executor.ainvoke({"input": task, "chat_history": []})
    return {
        "past_steps": (task, agent_response["agent_outcome"].return_values["output"])
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"objective": state["input"]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output, Response):
        return {"response": output.response}
    else:
        return {"plan": output.steps}


def should_end(state: PlanExecute):
    if state["response"]:
        return True
    else:
        return False
from langgraph.graph import StateGraph, END

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    {
        # If `tools`, then we call the tool node.
        True: END,
        False: "agent",
    },
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
from langchain_core.messages import HumanMessage

config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}

async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)

# Conclusion
# Congrats on making a plan-and-execute agent! 
# One known limitations of the above design is that each task is still executed in sequence, meaning embarassingly parallel operations all add to the total execution time. 
# You could improve on this by having each task represented as a DAG (similar to LLMCompiler), rather than a regular list.