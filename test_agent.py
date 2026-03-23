"""End-to-end agent test for OfficeQA with rollout logging to OpenReward."""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward, OpenReward
from openreward.api.environments.types import ToolCallError
from openreward.api.rollouts.serializers.base import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


async def main():
    async_or = AsyncOpenReward()
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/OfficeQA"
    SPLIT = "train"
    OR_API_KEY = os.getenv("OPENREWARD_API_KEY")

    environment = async_or.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    # Test first task
    for task in tasks[:1]:
        task_id = task.task_spec["id"]
        print(f"\n--- Running task: {task_id} ---")

        or_rollout = or_client.rollout.create(
            run_name="officeqa_test",
            rollout_name=f"test_{task_id}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
            print_messages=False,
        )

        async with environment.session(
            task=task,
            secrets={
                "OPENREWARD_API_KEY": (OR_API_KEY, ["api.openreward.ai", "sessions.openreward.ai"]),
            },
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            or_rollout.log(message=UserMessage(content=prompt[0].text))

            tool_call_count = 0
            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                input_list += response.output
                has_tool_calls = False

                for item in response.output:
                    if item.type == "function_call":
                        has_tool_calls = True
                        tool_call_count += 1
                        print(f"Tool #{tool_call_count}: {item.name}")

                        or_rollout.log(
                            message=ToolCall(
                                name=item.name,
                                content=item.arguments,
                                call_id=item.call_id,
                            )
                        )

                        try:
                            tool_result = await session.call_tool(
                                item.name,
                                json.loads(str(item.arguments)),
                            )
                            reward = tool_result.reward
                            finished = tool_result.finished
                            result_text = tool_result.blocks[0].text
                        except ToolCallError as e:
                            print(f"Tool error: {e}")
                            reward = 0.0
                            finished = False
                            result_text = f"Error: {e}"

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result_text,
                        })

                        or_rollout.log(
                            message=ToolResult(
                                content=result_text,
                                call_id=item.call_id,
                            ),
                            reward=reward,
                            is_finished=finished,
                        )

                        print(f"Reward: {reward:.4f} | Finished: {finished}")

                        if finished:
                            break

                    elif hasattr(item, "text") and item.text:
                        or_rollout.log(message=AssistantMessage(content=item.text))

                if not has_tool_calls:
                    print("Model responded with text (no tool calls), stopping.")
                    break

    print(f"\nTest complete ({tool_call_count} tool calls)")


if __name__ == "__main__":
    asyncio.run(main())
