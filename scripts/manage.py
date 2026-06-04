"""
岗位管理 CLI 工具

用法:
    python scripts/manage.py list-jobs
    python scripts/manage.py add-job --name "产品经理" --input "data/题库/产品经理.pdf" "data/题库/面经2.pdf"
    python scripts/manage.py add-job --name "产品经理" --input "data/题库/"          (整个目录)
    python scripts/manage.py mine --job-type "产品经理" --input "data/题库/产品经理.pdf" "data/题库/面经2.pdf"
    python scripts/manage.py build-index --job-type "产品经理"
    python scripts/manage.py rebuild --job-type "新媒体运营"
    python scripts/manage.py remove-job --job-type "产品经理"
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging

from app.config import settings
from domain.job_configs import load_job_config, list_available_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cmd_list_jobs(args):
    available_jobs = list_available_jobs()
    jobs_data_dir = settings.get_jobs_data_dir()

    if not available_jobs:
        print("没有找到任何岗位配置。")
        print(f"   请在 domain/job_configs/ 下创建 YAML 配置文件。")
        return

    print(f"\n{'='*60}")
    print(f"  已配置岗位列表")
    print(f"{'='*60}\n")

    for job_type in available_jobs:
        try:
            config = load_job_config(job_type)
        except Exception as e:
            print(f"  X {job_type} - 配置加载失败: {e}")
            continue

        job_dir = settings.get_job_data_dir(job_type)
        has_raw = (settings.get_job_raw_dir(job_type).exists() and
                   any(settings.get_job_raw_dir(job_type).iterdir()))
        has_processed = (settings.get_job_processed_dir(job_type).exists() and
                         (settings.get_job_processed_dir(job_type) / "mined_questions.jsonl").exists())
        has_index = (settings.get_job_index_dir(job_type).exists() and
                     (settings.get_job_index_dir(job_type) / "faiss_index").exists() and
                     (settings.get_job_index_dir(job_type) / "bm25.json").exists())

        status_parts = []
        status_parts.append("[raw] OK" if has_raw else "[raw] --")
        status_parts.append("[processed] OK" if has_processed else "[processed] --")
        status_parts.append("[index] OK" if has_index else "[index] --")

        status_str = " | ".join(status_parts)

        print(f"  {config.display_name} ({job_type})")
        print(f"     Status: {status_str}")
        print(f"     Dir:    {job_dir}")
        print()

    print(f"{'='*60}\n")


def cmd_add_job(args):
    from scripts.mine_textbook import mine_textbook_to_questions
    from scripts.build_index import build_and_save_index

    job_type = args.name
    input_paths = args.input

    try:
        config = load_job_config(job_type)
        logger.info(f"Found job config: {job_type}")
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        print(f"  Please create the config file first.")
        return

    raw_dir = settings.get_job_raw_dir(job_type)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for p in input_paths:
        if not Path(p).exists():
            print(f"X Input path does not exist: {p}")
            return

    print(f"\n{'='*60}")
    print(f"  Add Job: {config.display_name} ({job_type})")
    print(f"  Input:   {', '.join(input_paths)}")
    print(f"  Model:   {settings.OFFLINE_MODEL_NAME} (offline)")
    print(f"{'='*60}\n")

    print("Step 1/3: Extract text from files...")
    print("Step 2/3: LLM mining, extracting interview questions...")
    output_path = str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")
    mine_textbook_to_questions(input_paths, output_path, job_type=job_type)

    print("Step 3/3: Build search index...")
    build_and_save_index(job_type)

    print(f"\n{'='*60}")
    print(f"  Done! Job {config.display_name} added successfully!")
    print(f"{'='*60}\n")


def cmd_mine(args):
    from scripts.mine_textbook import mine_textbook_to_questions

    job_type = args.job_type
    input_paths = args.input

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    output_path = args.output or str(settings.get_job_processed_dir(job_type) / "mined_questions.jsonl")

    print(f"Mining: {config.display_name} ({job_type})")
    print(f"Input:  {', '.join(input_paths)}")
    print(f"Model:  {settings.OFFLINE_MODEL_NAME} (offline)")
    mine_textbook_to_questions(input_paths, output_path, job_type=job_type)


def cmd_build_index(args):
    from scripts.build_index import build_and_save_index

    job_type = args.job_type

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    print(f"Building index: {config.display_name} ({job_type})")
    build_and_save_index(job_type)


def cmd_rebuild(args):
    from scripts.build_index import build_and_save_index

    job_type = args.job_type

    try:
        config = load_job_config(job_type)
    except FileNotFoundError:
        print(f"X Job config not found: domain/job_configs/{job_type}.yaml")
        return

    index_dir = settings.get_job_index_dir(job_type)
    if index_dir.exists():
        import shutil
        shutil.rmtree(str(index_dir))
        logger.info(f"Deleted old index: {index_dir}")

    print(f"Rebuilding index: {config.display_name} ({job_type})")
    build_and_save_index(job_type)


def cmd_remove_job(args):
    import shutil

    job_type = args.job_type

    job_dir = settings.get_job_data_dir(job_type)
    if not job_dir.exists():
        print(f"X Job data directory not found: {job_dir}")
        return

    if not args.force:
        confirm = input(f"Confirm delete all data for '{job_type}'? (y/N): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    shutil.rmtree(str(job_dir))
    print(f"Deleted job data: {job_dir}")
    print(f"  Note: YAML config domain/job_configs/{job_type}.yaml needs to be deleted manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Interview Coach - Job Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage.py list-jobs
  python scripts/manage.py add-job --name "产品经理" --input "data/题库/产品经理.pdf"
  python scripts/manage.py add-job --name "产品经理" --input "data/题库/1.pdf" "data/题库/2.pdf" "data/题库/3.txt"
  python scripts/manage.py add-job --name "产品经理" --input "data/题库/"
  python scripts/manage.py mine --job-type "产品经理" --input "data/题库/1.pdf" "data/题库/2.pdf"
  python scripts/manage.py build-index --job-type "产品经理"
  python scripts/manage.py rebuild --job-type "新媒体运营"
  python scripts/manage.py remove-job --job-type "产品经理" --force
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    list_parser = subparsers.add_parser("list-jobs", help="List all configured jobs and their status")
    list_parser.set_defaults(func=cmd_list_jobs)

    add_parser = subparsers.add_parser("add-job", help="Add a new job (extract -> mine -> build index)")
    add_parser.add_argument("--name", "-n", required=True, help="Job name (must match YAML config filename)")
    add_parser.add_argument("--input", "-i", nargs="+", required=True,
                            help="Input file paths (PDF/TXT), supports multiple files and directories")
    add_parser.set_defaults(func=cmd_add_job)

    mine_parser = subparsers.add_parser("mine", help="Run the mining step only")
    mine_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    mine_parser.add_argument("--input", "-i", nargs="+", required=True,
                            help="Input file paths (PDF/TXT), supports multiple files and directories")
    mine_parser.add_argument("--output", "-o", default=None, help="Output path")
    mine_parser.set_defaults(func=cmd_mine)

    build_parser = subparsers.add_parser("build-index", help="Run the index building step only")
    build_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    build_parser.set_defaults(func=cmd_build_index)

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild index for a job")
    rebuild_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    rebuild_parser.set_defaults(func=cmd_rebuild)

    remove_parser = subparsers.add_parser("remove-job", help="Remove a job's data")
    remove_parser.add_argument("--job-type", "-j", required=True, help="Job type")
    remove_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    remove_parser.set_defaults(func=cmd_remove_job)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
