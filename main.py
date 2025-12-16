import os
import asyncio
import datetime
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

# 确保导入环境变量
from dotenv import load_dotenv
load_dotenv()

# --- 配置部分 ---
# Autogen 默认查找 OPENAI_API_KEY 环境变量

# 示例 1: 使用标准的 OpenAI API
config_list_openrouter = [
    {
        "model": "deepseek/deepseek-v3.2:online",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1", # 默认可以省略
    },
    {
        "model": "google/gemini-3-pro-preview:online",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1", # 默认可以省略
    }
]


# 选择您要使用的配置列表
config_list = config_list_openrouter

# 检查配置是否有效
if not config_list or not config_list[0].get("api_key") and "openai.com" in config_list[0].get("base_url", "openai.com"):
    print("警告：OPENAI_API_KEY 环境变量未设置，或配置列表为空。请检查您的 .env 文件或配置。")


# --- Autogen 智能体配置 (使用上一条回复中的泛化角色) ---

llm_config_for_agents = {
    "config_list": config_list,
    # 讨论场景下，设置温度可以鼓励更多样化的观点
    "temperature": 0.5, 
}

# 为不同智能体创建不同的模型客户端
model_info_deepseek = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="deepseek",
    structured_output=True,
)

model_info_gemini = ModelInfo(
    vision=True,
    function_calling=True,
    json_output=True,
    family="gemini",
    structured_output=True,
)


model_client_deepseek = OpenAIChatCompletionClient(
    model=config_list[0]["model"],
    api_key=config_list[0]["api_key"],
    base_url=config_list[0]["base_url"],
    model_info=model_info_deepseek,
    temperature=0.5,
)

model_client_gemini = OpenAIChatCompletionClient(
    model=config_list[1]["model"],
    api_key=config_list[1]["api_key"],
    base_url=config_list[1]["base_url"],
    model_info=model_info_gemini,
    temperature=0.2,
)

analyst = AssistantAgent(
    name="Analyst",
    model_client=model_client_deepseek,
    system_message="""你是一位中立的『分析师』。你的职责是：
1. 根据用户议题，提供相关背景信息、关键定义和客观数据对比
2. 确保讨论建立在坚实的事实基础上
3. 每次只回答当前问题，保持简洁（200-300字）
4. 如果需要补充信息，等待Manager的提问再回答
5. 不要一次性输出所有内容"""
)

advocate = AssistantAgent(
    name="Advocate",
    model_client=model_client_deepseek,
    system_message="""你是一位充满热情的『拥护者』。你的职责是：
1. 选择你认为最理想的方案，提出支持论据
2. 每次发言针对当前讨论焦点，不要一次性总结所有观点
3. 对Critic的质疑进行有力反驳
4. 保持简洁有力（200-300字）
5. 如果Critic还有其他质疑，等待下一轮讨论"""
)

critic = AssistantAgent(
    name="Critic",
    model_client=model_client_deepseek,
    system_message="""你是一位严谨的『批评家』。你的职责是：
1. 质疑Advocate的选择，指出缺点、风险和挑战
2. 每次发言针对一个主要问题，不要一次性列出所有批评
3. 根据Advocate的回应继续深入质疑
4. 保持简洁锐利（200-300字）
5. 只有在多轮讨论后各方观点充分交锋，再考虑是否达成共识"""
)

manager = AssistantAgent(
    name="Manager",
    model_client=model_client_gemini,
    system_message="""你是讨论主持人。你的职责是：
1. 引导讨论按顺序进行：Analyst提供背景 -> Advocate提出方案 -> Critic质疑 -> 反复交锋
2. 每次只提一个问题或引导一个讨论点，让各方逐步深入
3. 鼓励Advocate和Critic进行多轮交锋，至少5-10轮对话
4. 当发现讨论还不够充分时，继续提出新的角度让各方探讨
5. 只有当以下条件都满足时，才输出'FINAL REPORT'并总结：
   - Advocate和Critic已经进行了充分的多轮辩论（至少10轮）
   - 各方观点已经深入探讨，没有新的重要论点
   - 你认为已经可以形成全面的结论
6. 在未达成以上条件前，绝不要提及'FINAL REPORT'这个词
7. 每次发言保持简洁（100-200字），只引导下一个讨论点""",
)

user_proxy = UserProxyAgent(
    name="User",
    description="A user who provides the initial task and receives the final report",
)

# 4. 启动讨论 (任何议题都可以)
topic = "CPO、LPO和NPO的对比，并探讨未来柜内光互连最可能采用哪种方案"

# 用户选项：是否开启命令行即时输出每个对话内容
enable_realtime_output = True  # 设置为 False 可关闭实时输出

# 3. 设置群组聊天
termination_condition = TextMentionTermination("FINAL REPORT")

team = RoundRobinGroupChat(
    participants=[manager, analyst, advocate, critic, user_proxy],
    termination_condition=termination_condition,
    max_turns=100
)

async def main():
    print(f"\n--- 启动多智能体讨论：{topic} ---")
    print("="*80)

    # 使用流式输出来实时显示讨论过程
    messages = []
    message_count = 0
    
    async for message in team.run_stream(task=topic):
        messages.append(message)
        message_count += 1
        
        # 实时打印每条消息
        if enable_realtime_output:
            print(f"\n{'='*80}")
            print(f"第 {message_count} 轮 - [{message.source}]")
            print(f"{'='*80}")
            print(f"{message.content}")
            print(f"{'='*80}\n")

    print("\n--- 讨论完成 ---")
    
    # 生成时间戳
    timestamp_full = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.datetime.now().strftime("%Y%m%d")

    # 解析时间部分为时分秒
    time_str = timestamp_full.split('_')[1]
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    print(f"时间戳解析: 小时={hours}, 分钟={minutes}, 秒={seconds}")

    # 保存完整的讨论过程（包含所有agent的发言）
    discussion_content = "# 讨论过程记录\n\n"
    discussion_content += f"**议题**: {topic}\n\n"
    discussion_content += f"**时间**: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
    discussion_content += f"**总轮次**: {len(messages)} 条消息\n\n"
    discussion_content += "---\n\n"
    
    for i, msg in enumerate(messages, 1):
        discussion_content += f"### 第 {i} 轮 - {msg.source}\n\n{msg.content}\n\n---\n\n"

    filename_discussion = f"讨论过程_{timestamp_full}_{topic}.md"
    with open(filename_discussion, 'w', encoding='utf-8') as f:
        f.write(discussion_content)
    print(f"讨论过程已保存到文件：{filename_discussion}")

    # 提取并保存最终报告（查找包含FINAL REPORT的消息）
    final_report = None
    for msg in reversed(messages):
        if "FINAL REPORT" in msg.content:
            final_report = msg.content
            break
    
    if final_report:
        # 只保存FINAL REPORT部分（如果有分隔符的话）
        report_content = "# 最终报告\n\n"
        report_content += f"**议题**: {topic}\n\n"
        report_content += f"**时间**: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
        report_content += "---\n\n"
        report_content += final_report
        
        filename_report = f"{timestamp_full}_{topic}.md"
        with open(filename_report, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"最终报告已保存到文件：{filename_report}")
    else:
        print("警告：未找到包含'FINAL REPORT'的消息")

if __name__ == "__main__":
    asyncio.run(main())
