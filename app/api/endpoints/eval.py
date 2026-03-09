from fastapi import APIRouter, BackgroundTasks, HTTPException
import json
import os
import sys
import subprocess
import asyncio

# 把上级目录加入 sys.path, 因为 evaluation 包不在后端根目录，而是在 backend.evaluation
# 虽然可以直接访问相对路径，但复用 generate_report 的逻辑更好
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.generate_report import (
    load_result_safe,
    extract_summary,
    extract_details,
    normalize_retrieval,
    normalize_generation,
    normalize_coaching,
    normalize_citation,
    normalize_performance,
    find_low_score_cases,
    generate_optimization_suggestions
)

router = APIRouter()

# ─── 评测状态类 ─────────────────────────────────────────────────────────────
class EvalStatus:
    is_running = False
    current_module = ""
    progress = 0
    total_modules = 6
    error = None

status_store = EvalStatus()

@router.get("/summary")
async def get_evaluation_summary():
    """读取所有评测结果并返回给前端 Dashboard"""
    # 重新取一遍最新的结果数据
    retrieval_data   = load_result_safe("retrieval_results.json")
    ragas_data       = load_result_safe("ragas_results.json")
    coaching_data    = load_result_safe("coaching_results.json")
    citation_data    = load_result_safe("citation_results.json")
    performance_data = load_result_safe("performance_results.json")

    retrieval_summary   = extract_summary(retrieval_data)
    generation_summary  = extract_summary(ragas_data)
    coaching_summary    = extract_summary(coaching_data)
    citation_summary    = extract_summary(citation_data)
    performance_summary = extract_summary(performance_data)
    coaching_details    = extract_details(coaching_data)

    scores = {
        "检索质量": normalize_retrieval(retrieval_summary),
        "生成质量": normalize_generation(generation_summary),
        "教练质量": normalize_coaching(coaching_summary),
        "引用质量": normalize_citation(citation_summary),
        "性能表现": normalize_performance(performance_summary),
    }

    # 低分案例
    low_score_cases = find_low_score_cases(coaching_details, threshold=2.5)

    # 优化建议
    suggestions = generate_optimization_suggestions(
        retrieval_summary, generation_summary, coaching_summary,
        citation_summary, performance_summary,
    )

    return {
        "scores": scores,
        "summaries": {
            "retrieval": retrieval_summary,
            "generation": generation_summary,
            "coaching": coaching_summary,
            "citation": citation_summary,
            "performance": performance_summary,
        },
        "low_score_cases": low_score_cases[:5], # 最多前5条
        "suggestions": suggestions
    }

# ─── 自动化评测触发端点 ────────────────────────────────────────────────────

def run_eval_scripts_background():
    """在后台运行所有评测脚本"""
    scripts = [
        ("Retrieval 检索质量", "run_retrieval_eval.py"),
        ("RAGAS 生成质量", "run_ragas_eval.py"),
        ("Coaching 教练核心", "run_coaching_eval.py"),
        ("Citation 引用验证", "run_citation_eval.py"),
        ("Performance 性能打点", "run_performance_eval.py"),
        ("Report 报告生成", "generate_report.py")
    ]
    status_store.total_modules = len(scripts)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    eval_dir = os.path.join(base_dir, "evaluation")
    
    try:
        for i, (name, script) in enumerate(scripts):
            status_store.current_module = name
            status_store.progress = i
            
            script_path = os.path.join(eval_dir, script)
            cmd = ["uv", "run", "python", script_path]
            
            # 为了评测速度，性能测试限定为5个样本
            if script == "run_performance_eval.py":
                cmd.extend(["--samples", "5"])
                
            print(f"🏃 正在执行后台评测: {name} -> {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ {name} 执行失败:\n{result.stderr}")
                status_store.error = f"模块 {name} 运行中发生错误"
                status_store.is_running = False
                return
                
        status_store.progress = len(scripts)
        status_store.current_module = "✅ 评测已全部完成"
    except Exception as e:
        status_store.error = str(e)
    finally:
        status_store.is_running = False

@router.post("/run")
async def start_evaluation(background_tasks: BackgroundTasks):
    """触发全自动评测流水线"""
    if status_store.is_running:
        raise HTTPException(status_code=400, detail="评测流程已在运行中，请等待完成。")
        
    status_store.is_running = True
    status_store.current_module = "正在初始化..."
    status_store.progress = 0
    status_store.error = None
    
    background_tasks.add_task(run_eval_scripts_background)
    return {"message": "评测任务已在后台启动"}

@router.get("/status")
async def get_evaluation_status():
    """轮询当前评测进度"""
    return {
        "is_running": status_store.is_running,
        "current_module": status_store.current_module,
        "progress": status_store.progress,
        "total_modules": status_store.total_modules,
        "error": status_store.error
    }

