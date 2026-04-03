import json
import asyncio
import os
from pathlib import Path
from sqlmodel import select
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.database import get_session
from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt

EVAL_SYSTEM_PROMPT = """你是一个专业的 RAG (检索增强生成) 系统评估裁判。
你需要根据提供的 1) 问题, 2) 标准答案(Ground Truth), 3) 检索到的上下文, 4) AI生成的回答。
来评估系统的两项核心指标，并输出 JSON 格式的结果，包含 0-10 分的具体得分和简短的打分理由。

1. Context Precision (上下文精准度 0-10): 检索到的上下文是否包含了回答该问题所需的核心知识？如果上下文中混杂了大量无关内容，请扣分。
2. Answer Relevance & Faithfulness (回答相关性与忠实度 0-10): AI的回答是否直接且准确地回答了问题？是否紧扣标准答案的核心意思？有没有捏造上下文中不存在的（幻觉）内容？

请严格输出以下 JSON 格式：
{
    "context_precision_score": 8,
    "context_precision_reason": "理由...",
    "answer_score": 9,
    "answer_reason": "理由..."
}
"""

async def run_evaluation():
    # 1. 读取测试集
    dataset_path = Path(__file__).parent / "dataset" / "test_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    
    # 2. 初始化裁判大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    judge_llm = ChatTongyi(model="qwen-turbo", dashscope_api_key=api_key)
    student_llm = ChatTongyi(model="qwen-turbo", dashscope_api_key=api_key)
    
    results = []
    
    print("🚀 开始运行 RAG 系统评估框架...")
    
    async for db in get_session():
        for idx, item in enumerate(dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]
            print(f"\n[{idx+1}/{len(dataset)}] 正在测试问题: {question}")
            
            # --- RAG 流程模拟 ---
            # 1. 检索
            chunks = await retrieve_relevant_chunks(question, db)
            contexts = "\n---\n".join([c[0] for c in chunks])
            
            # 2. 生成回答
            student_prompt = build_rag_prompt(chunks, today_diary=None, today_date="2026-04-01", today_mood=None)
            response = student_llm.invoke([
                SystemMessage(content=student_prompt),
                HumanMessage(content=question)
            ])
            model_answer = response.content
            
            # --- 裁判打分 ---
            judge_input = f"""
【评估任务】
问题: {question}
标准答案(Ground Truth): {ground_truth}

【RAG 系统产出】
检索到的上下文:
{contexts if contexts else "未检索到内容"}

AI教练的回答:
{model_answer}
"""
            judge_response = judge_llm.invoke([
                SystemMessage(content=EVAL_SYSTEM_PROMPT),
                HumanMessage(content=judge_input)
            ])
            
            try:
                # 提取 JSON
                json_text = judge_response.content
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                score = json.loads(json_text)
                print(f"  👉 检索得分: {score['context_precision_score']}/10 - {score['context_precision_reason']}")
                print(f"  👉 回答得分: {score['answer_score']}/10 - {score['answer_reason']}")
                
                results.append(score)
            except Exception as e:
                print(f"  ⚠️ 解析打分结果失败: {e}")
                
    if results:
        avg_context = sum(r['context_precision_score'] for r in results) / len(results)
        avg_answer = sum(r['answer_score'] for r in results) / len(results)
        print("\n" + "="*40)
        print("🎯 最终评估报告")
        print("="*40)
        print(f"总计测试用例: {len(results)}")
        print(f"平均检索质量 (Context Quality): {avg_context:.1f} / 10.0")
        print(f"平均回答质量 (Answer Quality):  {avg_answer:.1f} / 10.0")
        print("="*40)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    asyncio.run(run_evaluation())
