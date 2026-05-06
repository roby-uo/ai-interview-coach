#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目文档生成器
自动扫描项目结构并生成架构图和代码文档
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()
    OUTPUT_FILE: str = "PROJECT_DOCUMENTATION.md"
    
    EXCLUDE_DIRS: Set[str] = field(default_factory=lambda: {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".chainlit", ".files", ".vscode", ".idea", ".pytest_cache",
        "dist", "build", "*.egg-info"
    })
    
    EXCLUDE_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        ".json", ".jsonl", ".toml", ".txt", ".pkl", ".faiss",
        ".pyc", ".pyo", ".bat", ".md", ".css", ".pdf", ".png", 
        ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".lock", ".log"
    })
    
    INCLUDE_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        ".py"
    })
    
    EXCLUDE_FILES: Set[str] = field(default_factory=lambda: {
        ".env", ".gitignore", ".python-version", ".env.example",
        "PROJECT_DOCUMENTATION.md"
    })
    
    SENSITIVE_PATTERNS: List[str] = field(default_factory=lambda: [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'OPENAI_API_KEY\s*=\s*["\'][^"\']+["\']',
    ])


config = Config()


def sanitize_content(content: str, file_path: Path) -> str:
    for pattern in config.SENSITIVE_PATTERNS:
        content = re.sub(pattern, '***REDACTED***', content, flags=re.IGNORECASE)
    return content


def should_exclude_dir(dir_name: str) -> bool:
    if dir_name in config.EXCLUDE_DIRS:
        return True
    for pattern in config.EXCLUDE_DIRS:
        if '*' in pattern and re.match(pattern.replace('*', '.*'), dir_name):
            return True
    return False


def should_include_file(file_name: str) -> bool:
    if file_name in config.EXCLUDE_FILES:
        return False
    
    ext = Path(file_name).suffix.lower()
    return ext in config.INCLUDE_EXTENSIONS


class TreeGenerator:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.output_lines = []
    
    def generate(self) -> str:
        self.output_lines = [f"{self.root_path.name}/"]
        self._generate_recursive(self.root_path, "", True)
        return "\n".join(self.output_lines)
    
    def _generate_recursive(self, current_path: Path, prefix: str, is_last: bool):
        try:
            items = sorted([
                item for item in current_path.iterdir()
                if not (item.is_dir() and should_exclude_dir(item.name))
                and item.name not in config.EXCLUDE_FILES
            ], key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "
            
            if item.is_dir():
                self.output_lines.append(f"{prefix}{connector}{item.name}/")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                self._generate_recursive(item, new_prefix, is_last_item)
            else:
                self.output_lines.append(f"{prefix}{connector}{item.name}")


class CodePackager:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.files: List[Path] = []
    
    def scan(self) -> List[Path]:
        self.files = []
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
            for file in files:
                if should_include_file(file):
                    file_path = Path(root) / file
                    if file_path.name not in config.EXCLUDE_FILES:
                        self.files.append(file_path)
        self.files.sort()
        return self.files
    
    def get_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return sanitize_content(content, file_path)
        except UnicodeDecodeError:
            return "[Binary file - cannot display]"
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def get_file_stats(self, file_path: Path) -> Dict:
        content = self.get_file_content(file_path)
        lines = content.split('\n') if not content.startswith('[') else []
        return {
            'lines': len(lines),
            'size': file_path.stat().st_size if file_path.exists() else 0
        }


class DocumentationGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.tree_generator = TreeGenerator(config.PROJECT_ROOT)
        self.code_packager = CodePackager(config.PROJECT_ROOT)
    
    def generate(self) -> str:
        sections = []
        
        sections.extend(self._generate_header())
        sections.extend(self._generate_toc())
        sections.extend(self._generate_architecture())
        sections.extend(self._generate_code_section())
        sections.extend(self._generate_footer())
        
        return "\n".join(sections)
    
    def _generate_header(self) -> List[str]:
        return [
            "# 面试教练项目文档",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 项目路径: `{self.config.PROJECT_ROOT}`",
            "",
            "---",
            ""
        ]
    
    def _generate_toc(self) -> List[str]:
        return [
            "## 目录",
            "",
            "- [项目架构](#项目架构)",
            "- [核心代码](#核心代码)",
            "  - [应用层 (app/)](#应用层-app)",
            "  - [核心层 (core/)](#核心层-core)",
            "  - [数据层 (data/)](#数据层-data)",
            "  - [领域层 (domain/)](#领域层-domain)",
            "  - [基础设施层 (infrastructure/)](#基础设施层-infrastructure)",
            "  - [脚本 (scripts/)](#脚本-scripts)",
            "- [统计信息](#统计信息)",
            "",
            "---",
            ""
        ]
    
    def _generate_architecture(self) -> List[str]:
        tree = self.tree_generator.generate()
        return [
            "## 项目架构",
            "",
            "```",
            tree,
            "```",
            "",
            "### 架构说明",
            "",
            "| 目录 | 说明 |",
            "|------|------|",
            "| `app/` | 应用入口和配置（Chainlit UI层） |",
            "| `core/` | 核心业务逻辑（LangGraph图、节点、技能、用户画像） |",
            "| `core/utils/` | 工具模块（LLM工厂、通用工具函数） |",
            "| `data/` | 数据层（向量索引、处理后数据、原始数据） |",
            "| `domain/` | 领域模型、数据结构和业务验证 |",
            "| `domain/job_configs/` | 岗位配置文件（YAML格式，支持多岗位） |",
            "| `infrastructure/` | 基础设施（解析器、检索器、工具） |",
            "| `scripts/` | 构建索引、数据处理脚本、CLI管理工具 |",
            "| `public/` | Chainlit静态资源（CSS等） |",
            "| `.chainlit/` | Chainlit配置 |",
            "",
            "---",
            ""
        ]
    
    def _generate_code_section(self) -> List[str]:
        files = self.code_packager.scan()
        sections = ["## 核心代码\n"]
        
        grouped_files = self._group_files_by_directory(files)
        
        dir_order = [
            ('app', '应用层 (app/)', 'Chainlit应用入口和配置'),
            ('core', '核心层 (core/)', 'LangGraph图定义、节点实现、技能模块、用户画像'),
            ('data', '数据层 (data/)', '向量索引、处理后数据、原始数据'),
            ('domain', '领域层 (domain/)', '数据模型、业务实体和验证规则'),
            ('infrastructure', '基础设施层 (infrastructure/)', '文件解析、向量检索、工具集成'),
            ('scripts', '脚本 (scripts/)', '构建索引和数据处理脚本'),
        ]
        
        for dir_name, title, description in dir_order:
            if dir_name in grouped_files:
                sections.extend(self._generate_directory_section(
                    dir_name, title, description, grouped_files[dir_name]
                ))
        
        other_files = []
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            parts = rel_path.parts
            if parts[0] not in [d[0] for d in dir_order]:
                other_files.append(file_path)
        
        if other_files:
            sections.extend(self._generate_directory_section(
                'other', '其他文件', '根目录和其他文件', other_files
            ))
        
        sections.extend(self._generate_statistics(files))
        
        return sections
    
    def _group_files_by_directory(self, files: List[Path]) -> Dict[str, List[Path]]:
        grouped = {}
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            top_dir = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'
            if top_dir not in grouped:
                grouped[top_dir] = []
            grouped[top_dir].append(file_path)
        return grouped
    
    def _generate_directory_section(self, dir_name: str, title: str, 
                                     description: str, files: List[Path]) -> List[str]:
        sections = [
            f"### {title}",
            "",
            f"*{description}*",
            ""
        ]
        
        for file_path in files:
            rel_path = file_path.relative_to(self.config.PROJECT_ROOT)
            content = self.code_packager.get_file_content(file_path)
            
            sections.append(f"#### `{rel_path}`")
            sections.append("")
            sections.append("```python")
            sections.append(content)
            sections.append("```")
            sections.append("")
        
        return sections
    
    def _generate_statistics(self, files: List[Path]) -> List[str]:
        total_lines = 0
        total_size = 0
        file_stats = []
        
        for file_path in files:
            stats = self.code_packager.get_file_stats(file_path)
            total_lines += stats['lines']
            total_size += stats['size']
            file_stats.append((file_path, stats))
        
        return [
            "---",
            "",
            "## 统计信息",
            "",
            f"- **Python 文件数量**: {len(files)}",
            f"- **总代码行数**: {total_lines:,}",
            f"- **总文件大小**: {total_size / 1024:.1f} KB",
            ""
        ]
    
    def _generate_footer(self) -> List[str]:
        return [
            "---",
            "",
            "*本文档由自动化脚本生成，敏感信息已脱敏处理*",
            ""
        ]
    
    def save(self, output_path: Optional[Path] = None):
        content = self.generate()
        output = output_path or self.config.PROJECT_ROOT / self.config.OUTPUT_FILE
        
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        
        return output


def main():
    print("=" * 60)
    print("项目文档生成器")
    print("=" * 60)
    print()
    
    print(f"[1/3] 扫描项目目录: {config.PROJECT_ROOT}")
    generator = DocumentationGenerator(config)
    
    print("[2/3] 生成项目架构图...")
    tree = generator.tree_generator.generate()
    print(f"      发现 {len(tree.splitlines())} 个目录/文件")
    
    print("[3/3] 打包核心代码...")
    files = generator.code_packager.scan()
    print(f"      共 {len(files)} 个 Python 文件")
    
    output_path = generator.save()
    print()
    print("=" * 60)
    print(f"[完成] 文档已生成: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
