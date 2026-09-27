#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pypdf>=5.0",
#   "requests>=2.32",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from pypdf import PdfReader


SKILL_DIR = Path(__file__).resolve().parent
VAULT_DIR = SKILL_DIR.parent.parent
SOURCE_DIRS = (
    VAULT_DIR / "01_Sources" / "论文报告",
)
OUTPUT_DIR = VAULT_DIR / "05_Review" / "每日简报"
KNOWLEDGE_NOTE_DIR = VAULT_DIR / "02_Knowledge" / "模型与案例"
KNOWLEDGE_INDEX_FILE = VAULT_DIR / "02_Knowledge" / "README.md"
REFERENCE_PDF_DIR = VAULT_DIR / "01_Sources" / "论文报告" / "ReferencesPDF"
STATE_FILE = SKILL_DIR / "state.json"
ENV_FILE = Path.home() / "Documents" / "Horizon" / ".env"
CODEX_BIN = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
INTENSIVE_SKILL_FILE = Path(
    "/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/intensive-paper-reading/SKILL.md"
)
OBSIDIAN_MARKDOWN_SKILL_FILE = Path(
    "/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md"
)
OBSIDIAN_MARKDOWN_REFERENCE_FILES = (
    "/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/references/PROPERTIES.md",
    "/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/references/CALLOUTS.md",
    "/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/references/EMBEDS.md",
)


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_model_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)


def pick_pdf(target: date) -> Path:
    pdfs = sorted(
        path
        for source_dir in SOURCE_DIRS
        if source_dir.exists()
        for path in source_dir.glob("*.pdf")
    )
    if not pdfs:
        raise FileNotFoundError(
            "No PDF files found in 01_Sources/论文报告"
        )

    used: dict[str, str] = {}
    if STATE_FILE.exists():
        try:
            used = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            used = {}

    previous_date = used.get("date")
    previous_paper = used.get("paper")
    if previous_date == target.isoformat() and previous_paper:
        chosen = next((path for path in pdfs if path.name == previous_paper), None)
        return chosen or pdfs[0]

    if previous_paper:
        for index, candidate in enumerate(pdfs):
            if candidate.name == previous_paper:
                return pdfs[(index + 1) % len(pdfs)]
    return pdfs[0]


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    return clean_model_text("\n".join(chunks))


def locate_references(text: str) -> str:
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:references|bibliography|文献与参考|参考文献)\s*[:：]?\s*$",
            text,
        )
    )
    if not matches:
        matches = list(re.finditer(r"(?i)references", text))
    if not matches:
        raise ValueError("No References section found in PDF")
    return text[matches[-1].end():]


def parse_references(text: str) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?m)\s*\[(\d{1,3})\]\s*", normalized)
    entries: list[str] = []
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            entry = parts[i + 1] if i + 1 < len(parts) else ""
            entry = re.sub(r"\s+", " ", entry).strip()
            if len(entry) >= 30:
                entries.append(entry)
    else:
        blocks = re.split(r"(?m)(?=\b\d{1,3}\.\s+[A-Z])", normalized)
        entries = [re.sub(r"\s+", " ", block).strip() for block in blocks]
        entries = [entry for entry in entries if len(entry) >= 30]

    seen: set[str] = set()
    unique: list[str] = []
    for entry in entries:
        key = re.sub(r"[^a-z0-9]", "", entry.lower())[:120]
        if key not in seen:
            unique.append(entry)
            seen.add(key)
    return unique


def reference_year(entry: str) -> int | None:
    years = []
    for match in re.finditer(r"\b(19|20)\d{2}\b", entry):
        year = int(match.group(0))
        if 1900 <= year <= date.today().year + 1:
            years.append(year)
    return max(years) if years else None


def recent_references(entries: list[str], target: date, limit: int = 80) -> list[dict]:
    lower_bound = target.year - 4
    recent: list[dict] = []
    for entry in entries:
        year = reference_year(entry)
        if year is not None and lower_bound <= year <= target.year:
            recent.append({"year": year, "entry": entry})

    keywords = (
        "recommend", "recommender", "retrieval", "ranking", "generative",
        "language model", "llm", "semantic id", "tokeniz", "rerank",
    )
    recent.sort(
        key=lambda item: (
            not any(keyword in item["entry"].lower() for keyword in keywords),
            item["year"],
        ),
        reverse=False,
    )
    return recent[:limit]


def text_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "towards", "toward", "using",
        "based", "large", "language", "models", "model", "recommendation",
        "recommender", "systems", "system", "learning", "via", "through",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def entries_match(left: str, right: str) -> bool:
    import difflib

    left_norm = re.sub(r"[^a-z0-9]", "", left.lower())
    right_norm = re.sub(r"[^a-z0-9]", "", right.lower())
    if left_norm and right_norm:
        ratio = difflib.SequenceMatcher(None, left_norm[:360], right_norm[:360]).ratio()
        if ratio >= 0.58:
            return True

    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.42


def build_reference_index() -> list[dict]:
    indexed: list[dict] = []
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue
        for pdf_path in sorted(source_dir.glob("*.pdf")):
            try:
                text = extract_pdf_text(pdf_path)
                reference_text = locate_references(text)
                paper_entries = parse_references(reference_text)
            except Exception:
                continue
            for entry in paper_entries:
                indexed.append({"source": pdf_path.name, "entry": entry})
    return indexed


def historical_reference_count(entry: str, reference_index: list[dict]) -> int:
    sources = {
        item["source"]
        for item in reference_index
        if entries_match(entry, item["entry"])
    }
    return len(sources)


def knowledge_note_titles() -> list[str]:
    if not KNOWLEDGE_NOTE_DIR.exists():
        return []
    return sorted(path.stem for path in KNOWLEDGE_NOTE_DIR.glob("*.md"))


def note_title_matches(reference_title: str, note_title: str) -> bool:
    acronym = re.split(r"[：:（(]", note_title, maxsplit=1)[0].strip()
    if len(acronym) >= 3 and re.search(
        rf"\b{re.escape(acronym)}\b", reference_title, flags=re.IGNORECASE
    ):
        return True
    return entries_match(reference_title, note_title)


def match_knowledge_note(reference_title: str, note_titles: list[str]) -> str | None:
    return next(
        (note_title for note_title in note_titles if note_title_matches(reference_title, note_title)),
        None,
    )


def annotate_recent_references(
    recent: list[dict],
    reference_index: list[dict],
    note_titles: list[str],
) -> list[dict]:
    for item in recent:
        item["historical_count"] = historical_reference_count(item["entry"], reference_index)
        item["already_deep_read_note"] = match_knowledge_note(item["entry"], note_titles)
    return recent


def call_model(
    source_paper: str,
    source_text: str,
    references: list[dict],
    note_titles: list[str],
) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    base_url = os.environ.get("PAPER_BRIEF_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("PAPER_BRIEF_MODEL", "deepseek-chat")
    first_pages = clean_model_text(source_text)[:6000]
    safe_references = clean_model_text(json.dumps(references, ensure_ascii=False, indent=2))
    safe_note_titles = clean_model_text(json.dumps(note_titles, ensure_ascii=False))
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是生成式推荐方向的研究助理。请只依据用户给出的论文文本和参考文献条目作答；"
                    "信息不足时明确写「信息不足」，不要编造作者、结论、数据或实验结果。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"源论文文件：{source_paper}\n"
                    f"源论文开头内容：\n{first_pages}\n\n"
                    f"以下是源论文 References 中近 5 年的候选条目：\n"
                    f"{safe_references}\n\n"
                    "请完成：\n"
                    "1. 提取源论文标题、短标题（不超过 20 字符）和一句话主题。\n"
                    "2. 从候选条目中选出最多 5 篇与生成式推荐、LLM 推荐、语义 ID、"
                    "生成式召回/排序、推荐评测最相关的论文。\n"
                    "3. 每篇给出核心思路、与源论文的关系、对生成式推荐研究的价值，"
                    "以及信息不足时需要人工核实的点。\n"
                    "4. 候选里的 historical_count 是它在本地底稿论文 References 中的去重被引数；"
                    "already_deep_read_note 是已经精读过的知识卡片标题。"
                    "如果 already_deep_read_note 为空但你确定某个候选对应下列卡片，"
                    "可以填写卡片标题；不确定就保持空。\n"
                    "5. 优先保留高历史被引和已有精读笔记的候选，但仍要与生成式推荐主题相关。\n"
                    "6. 只输出 JSON，字段为 paper_title、paper_short、source_summary、"
                    "selected_references。selected_references 中每项包含 "
                    "title/year/venue_or_link/relevance/historical_count/"
                    "already_deep_read_note/core_idea/relation_to_source/"
                    "value_for_research/caution。\n"
                    f"本地已有知识卡片标题：\n{safe_note_titles}"
                ),
            },
        ],
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Model API error {response.status_code}: {response.text[:500]}")
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "", text).strip()


def reference_count_label(item: dict) -> int:
    try:
        return max(0, int(item.get("historical_count", 0)))
    except (TypeError, ValueError):
        return 0


def find_recent_for_item(item: dict, recent: list[dict]) -> dict | None:
    title = str(item.get("title", ""))
    matches = [
        candidate
        for candidate in recent
        if entries_match(title, candidate["entry"])
    ]
    return matches[0] if matches else None


def deduplicate_selected_references(selected: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen_titles: set[str] = set()
    for item in selected:
        title_key = re.sub(
            r"[^a-z0-9]",
            "",
            str(item.get("title", "")).lower(),
        )[:180]
        if title_key in seen_titles:
            continue
        unique.append(item)
        seen_titles.add(title_key)
    return unique


def reconcile_selected_references(
    selected: list[dict],
    recent: list[dict],
    note_titles: list[str],
) -> list[dict]:
    for item in selected:
        candidate = find_recent_for_item(item, recent)
        if candidate:
            item["historical_count"] = reference_count_label(candidate)
            inherited_note = candidate.get("already_deep_read_note")
        else:
            item["historical_count"] = reference_count_label(item)
            inherited_note = None

        note = item.get("already_deep_read_note") or inherited_note
        if note not in note_titles:
            note = match_knowledge_note(str(item.get("title", "")), note_titles)
        item["already_deep_read_note"] = note
    return deduplicate_selected_references(selected)


def extract_arxiv_id(*texts: str) -> str | None:
    for text in texts:
        match = re.search(
            r"arxiv[:\s]*([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return None


def download_reference_pdf(item: dict) -> Path | None:
    arxiv_id = extract_arxiv_id(
        str(item.get("title", "")),
        str(item.get("venue_or_link", "")),
    )
    if not arxiv_id:
        return None

    title = safe_filename(str(item.get("title", f"arxiv-{arxiv_id}")))[:80]
    REFERENCE_PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REFERENCE_PDF_DIR / f"{title or arxiv_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        return pdf_path

    response = requests.get(
        f"https://arxiv.org/pdf/{arxiv_id}",
        timeout=60,
        headers={"User-Agent": "Obsidian-GenRec-Brief/1.0"},
    )
    if response.status_code >= 400 or len(response.content) < 10000:
        return None
    pdf_path.write_bytes(response.content)
    return pdf_path


def register_note_in_index(note_title: str, summary: str) -> None:
    if not KNOWLEDGE_INDEX_FILE.exists():
        return
    text = KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8")
    if f"[[{note_title}]]" in text:
        return

    section_match = re.search(r"^## 模型与案例\s*$", text, flags=re.MULTILINE)
    if not section_match:
        return
    section_start = section_match.end()
    next_section = re.search(r"^## ", text[section_start:], flags=re.MULTILINE)
    section_end = section_start + next_section.start() if next_section else len(text)
    section = text[section_start:section_end]
    table_rows = list(re.finditer(r"^\| .+\|\s*$", section, flags=re.MULTILINE))
    if not table_rows:
        return
    insert_at = section_start + table_rows[-1].end()
    row = f"\n| [[{note_title}]] | {summary} |"
    KNOWLEDGE_INDEX_FILE.write_text(text[:insert_at] + row + text[insert_at:], encoding="utf-8")


def format_note_with_obsidian_skill(note_path: Path) -> None:
    if not OBSIDIAN_MARKDOWN_SKILL_FILE.exists():
        raise RuntimeError(
            f"Obsidian Markdown skill not found: {OBSIDIAN_MARKDOWN_SKILL_FILE}"
        )
    if not CODEX_BIN.exists():
        raise RuntimeError(f"Codex CLI not found: {CODEX_BIN}")

    skill_text = OBSIDIAN_MARKDOWN_SKILL_FILE.read_text(encoding="utf-8")
    reference_texts = []
    for reference_file in OBSIDIAN_MARKDOWN_REFERENCE_FILES:
        reference_path = Path(reference_file)
        if reference_path.exists():
            reference_texts.append(reference_path.read_text(encoding="utf-8"))

    prompt = (
        "请严格执行下面的 obsidian-markdown skill，对一个已完成的 Markdown 文件执行格式化 pass。\n\n"
        "## Skill 全文\n\n"
        f"{skill_text}\n\n"
        "## 参考文件内容\n\n"
        + "\n\n".join(reference_texts)
        + "\n\n"
        "## 格式化要求\n\n"
        f"- 只允许编辑 {note_path}，不要读取或修改其它 Markdown 文件。\n"
        "- 只修正 Obsidian Markdown 格式；不得改变研究结论、方法描述、数字、来源、"
        "标签含义、状态或证据强度。\n"
        "- YAML frontmatter 必须合法；日期保持 YYYY-MM-DD；列表使用 YAML 列表；"
        "来源中的 PDF wikilink 使用引号包裹。\n"
        "- 知识库内部链接保持 [[wikilink]]；外部 URL 使用 [文本](https://...) Markdown 链接。\n"
        "- 保留或合理使用表格、代码块和标题；重要提醒可使用 > [!warning]、> [!info] 等"
        "Obsidian callout，但不要为了装饰而添加空话。\n"
        "- 状态必须保持「待复核」。\n"
        "- 不使用「待补充」「TODO」「详见原文」这类占位符。\n"
        "- 不修改创建日期；本次只是格式化，不要更新更新日期。\n"
        "- 完成后重新读取一次该文件，确认 YAML、表格、代码块和 callout 均可正常渲染。\n"
    )

    completed = subprocess.run(
        [
            str(CODEX_BIN),
            "exec",
            "--ignore-rules",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-C",
            str(VAULT_DIR),
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-1200:]
        raise RuntimeError(
            f"Obsidian Markdown formatting failed ({completed.returncode}): {detail}"
        )


def create_high_frequency_note(
    target: date,
    item: dict,
    source_paper: Path,
) -> str | None:
    pdf_path = download_reference_pdf(item)
    if not pdf_path:
        return None

    if not INTENSIVE_SKILL_FILE.exists():
        raise RuntimeError(f"Intensive reading skill not found: {INTENSIVE_SKILL_FILE}")
    if not CODEX_BIN.exists():
        raise RuntimeError(f"Codex CLI not found: {CODEX_BIN}")

    skill_text = INTENSIVE_SKILL_FILE.read_text(encoding="utf-8")
    result_path = SKILL_DIR / ".last_intensive_result.json"
    result_path.unlink(missing_ok=True)
    prompt = (
        "请严格执行下面的 intensive-paper-reading skill，完成一篇学术论文的结构化精读。\n\n"
        "## Skill 全文\n\n"
        f"{skill_text}\n\n"
        "## 本次任务\n\n"
        f"- 输入 PDF：{pdf_path}\n"
        f"- 发现来源：{source_paper.name}\n"
        f"- 运行日期：{target.isoformat()}\n"
        f"- 历史被引底稿论文数：{item.get('historical_count', 0)}\n"
        f"- 候选标题：{item.get('title', '')}\n"
        f"- 笔记目录：{KNOWLEDGE_NOTE_DIR}\n"
        f"- 知识卡索引：{KNOWLEDGE_INDEX_FILE}\n\n"
        "执行要求：\n"
        "1. 先按 skill 的筛选、速览、通读、精读流程处理 PDF，不要只读前几页。\n"
        "2. 在「模型与案例」目录创建或更新一张 Obsidian 知识卡片，文件名使用「主题 + 核心观点」。\n"
        "3. YAML 必须包含创建日期、更新日期、类型、标签、来源、状态、证据强度和待验证问题；"
        "状态使用「待复核」，来源使用指向 PDF 的 wikilink。\n"
        "4. 笔记必须保留 skill 的七问笔记、批判性分析、关键图表解读、值得追踪的引用、"
        "术语与句式积累和复现清单，并结合生成式推荐研究方向补充价值判断。\n"
        "5. 数字和实验结论必须写清场景/数据集、基线、指标和条件；不要编造。\n"
        "6. 不使用「待补充」「TODO」「详见原文」这类占位符；确实无法提取时写明原因。\n"
        "7. 将新卡片登记到知识卡索引的「模型与案例」表格；如果索引已有该卡片则不重复登记。\n"
        "8. 不要修改 01_Sources 中的底稿 PDF，也不要修改每日简报。\n"
        f"9. 最后把结果 JSON 写入 {result_path}，字段为 note_title、one_line、card_path、"
        "index_registered。note_title 必须与实际卡片文件名一致。\n"
    )

    completed = subprocess.run(
        [
            str(CODEX_BIN),
            "exec",
            "--ignore-rules",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-C",
            str(VAULT_DIR),
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=2400,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-1200:]
        raise RuntimeError(f"Intensive reading skill failed ({completed.returncode}): {detail}")
    if not result_path.exists():
        raise RuntimeError("Intensive reading skill did not write its result JSON")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    note_title = safe_filename(str(result.get("note_title", "")))[:90]
    note_path = KNOWLEDGE_NOTE_DIR / f"{note_title}.md"
    if not note_title or not note_path.exists():
        raise RuntimeError("Intensive reading skill returned an invalid note title")

    format_note_with_obsidian_skill(note_path)

    if not result.get("index_registered"):
        register_note_in_index(
            note_title,
            str(result.get("one_line", "历史高频引用论文精读")),
        )
    return note_title


def render_brief(target: date, source_paper: Path, result: dict, stats: dict) -> str:
    selected = result.get("selected_references", [])
    lines = [
        "---",
        f"创建日期: {target.isoformat()}",
        f"更新日期: {target.isoformat()}",
        "类型: 复盘",
        "标签:",
        "  - 生成式推荐",
        "  - 推荐系统",
        f'来源: "{source_paper.name}"',
        "---",
        "",
        f"# {target.isoformat()}｜论文引文简读",
        "",
        "## 底稿论文",
        "",
        f"- **论文**：{result.get('paper_title', source_paper.stem)}",
        f"- **一句话主题**：{result.get('source_summary', '信息不足')}",
        f"- **References 总量**：{stats['total']}",
        f"- **近 5 年候选**：{stats['recent']}",
        f"- **今日精选**：{len(selected)}",
        "",
        "## 精选引文",
        "",
    ]

    if not selected:
        lines.append("今天没有筛选出足够相关的近 5 年引文，建议换用下一篇论文再试。")
    for index, item in enumerate(selected, start=1):
        title = item.get("title", "未命名")
        year = item.get("year", "未知")
        venue = item.get("venue_or_link", "信息不足")
        lines.extend(
            [
                f"### {index}. {title}（{year}）",
                "",
                f"- **来源**：{venue}",
                f"- **相关性**：{item.get('relevance', '信息不足')}/5",
                f"- **核心思路**：{item.get('core_idea', '信息不足')}",
                f"- **与源论文的关系**：{item.get('relation_to_source', '信息不足')}",
                f"- **研究价值**：{item.get('value_for_research', '信息不足')}",
                f"- **历史被引**：{item.get('historical_count', 0)} 篇底稿论文",
            ]
        )
        existing_note = item.get("already_deep_read_note")
        if existing_note:
            lines.append(f"- **已有精读**：[[{existing_note}]]")
        elif reference_count_label(item) > 3:
            if item.get("deep_read_error"):
                lines.append("- **处理状态**：高频引用精读失败，简报已保留")
            else:
                lines.append("- **处理状态**：高频引用，已尝试生成精读笔记")
        else:
            lines.append("- **处理状态**：普通简读")
        lines.extend(
            [
                f"- **需要核实**：{item.get('caution', '信息不足')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 使用方式",
            "",
            "- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。",
            "- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。",
            "- 如某条内容信息不足，不要直接引用其结论，先查原文。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Run for a specific YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Parse references only")
    parser.add_argument("--force", action="store_true", help="Overwrite today's brief")
    args = parser.parse_args()

    load_env()
    tz = ZoneInfo("Asia/Shanghai")
    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now(tz).date()
    )
    source_paper = pick_pdf(target)
    note_titles = knowledge_note_titles()
    reference_index = build_reference_index()
    text = extract_pdf_text(source_paper)
    reference_text = locate_references(text)
    entries = parse_references(reference_text)
    recent = recent_references(entries, target)
    recent = annotate_recent_references(recent, reference_index, note_titles)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "date": target.isoformat(),
                    "paper": source_paper.name,
                    "references": len(entries),
                    "recent": len(recent),
                    "sample": recent[:3],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not recent:
        raise RuntimeError("No recent references found")

    result = call_model(source_paper.name, text, recent, note_titles)
    result["selected_references"] = reconcile_selected_references(
        result.get("selected_references", []),
        recent,
        note_titles,
    )

    for item in result["selected_references"]:
        if reference_count_label(item) > 3 and not item.get("already_deep_read_note"):
            try:
                note_title = create_high_frequency_note(target, item, source_paper)
                if note_title:
                    item["already_deep_read_note"] = note_title
            except Exception as exc:
                item["deep_read_error"] = str(exc)[:300]
    paper_short = safe_filename(str(result.get("paper_short") or source_paper.stem))[:28]
    output_path = OUTPUT_DIR / f"{target.isoformat()}｜论文引文简报：{paper_short}.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        print(f"Already exists: {output_path}")
        return

    content = render_brief(target, source_paper, result, {"total": len(entries), "recent": len(recent)})
    output_path.write_text(content, encoding="utf-8")
    format_note_with_obsidian_skill(output_path)
    STATE_FILE.write_text(
        json.dumps({"date": target.isoformat(), "paper": source_paper.name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
