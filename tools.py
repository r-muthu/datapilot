import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

# ----------------------- Workspace Manager -----------------------
class WorkspaceManager:
    def __init__(self):
        self.workspace: Optional[Path] = None
        self.original_path: Optional[Path] = None
        self.structure_cache = None
        self.sample_split_path = None

workspace_mgr = WorkspaceManager()

# Allowed parent directories (tune as needed)
ALLOWED_PARENTS = [Path.home() / "datasets", Path.home() / "Downloads", Path.home() / "projects"]

# def is_allowed_path(path: Path) -> bool:
#     for parent in ALLOWED_PARENTS:
#         if str(path.resolve()).startswith(str(parent.resolve())):
#             return True
#     return False

# ----------------------- Entry Tool -----------------------
@tool
def copy_the_dataset(source_path: str, should_copy: bool = True) -> str:
    """
    Initialise the workspace for the dataset.
    If copy=True (default), creates a timestamped copy in the same parent directory.
    If copy=False, sets the workspace directly to the original path (WARNING: modifications will be on original).
    Returns the workspace path, or "None" if the source is invalid.
    Must be called first before any other tools.
    """
    src = Path(source_path).resolve()
    if not src.exists() or not src.is_dir():
        return "None"

    if should_copy:
        parent = src.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ws_path = parent / f"{src.name}_copy_{timestamp}"
        shutil.copytree(src, ws_path)
        workspace_mgr.workspace = ws_path
        workspace_mgr.original_path = src
        return str(ws_path)
    else:
        workspace_mgr.workspace = src
        workspace_mgr.original_path = src
        return f"{src} (WARNING: working directly on original, modifications will be permanent)"

# ----------------------- Exploration Tools (no path argument) -----------------------
@tool
def explore_structure() -> str:
    """
    Return a JSON object describing the directory tree of the current workspace.
    """
    ds = workspace_mgr.workspace
    if not ds:
        return json.dumps({"error": "No workspace set. Call copy_dataset first."})
    tree = {}
    for root, dirs, files in os.walk(ds):
        rel = str(Path(root).relative_to(ds))
        if rel == ".":
            rel = "root"
        ext_counts = {}
        for f in files:
            ext = Path(f).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        tree[rel] = {"dirs": dirs, "files_count": len(files), "extensions": ext_counts}
    # Cache it
    workspace_mgr.structure_cache = tree
    return json.dumps(tree, indent=2)


@tool
def get_split_summary() -> str:
    """
    Scan the workspace cache to identify obvious dataset partitions (splits).
    Returns a quick directory list if standard splits exist, otherwise provides
    guidance for the agent to calculate splits manually via code execution.
    """
    ds = workspace_mgr.workspace
    structure = workspace_mgr.structure_cache

    if not ds:
        return json.dumps({"error": "No workspace set. Run 'copy_dataset' first."})
    if not structure:
        return json.dumps({"error": "Directory structure cache missing. Run 'explore_structure' first."})

    detected_splits = {}
    
    # Simple check for top-level folders matching split keywords
    split_keywords = {"train", "training", "val", "valid", "validation", "test", "testing", "dev"}
    
    for rel_path, info in structure.items():
        # Look only at top-level folders (direct children of root)
        if "/" not in rel_path and rel_path != "root":
            if any(keyword in rel_path.lower() for keyword in split_keywords):
                # Total files inside this split folder subtree
                detected_splits[rel_path] = info.get("files_count", 0)

    # Scenario A: Standard folder splits found
    if detected_splits:
        return json.dumps({
            "status": "success",
            "method": "directory_heuristic",
            "detected_folder_splits": detected_splits,
            "note": "Review file distributions. If these counts don't represent the true target split data, write a custom calculation script using run_python."
        }, indent=2)

    # Scenario B: Flat directory or complex structure (Fallback)
    all_extensions = set()
    for info in structure.values():
        all_extensions.update(info.get("extensions", {}).keys())

    return json.dumps({
        "status": "ambiguous",
        "reason": "No explicit top-level train/val/test folders found.",
        "instruction_for_agent": (
            "The dataset split configuration is flat, tabular, or custom. "
            f"Available file extensions in workspace: {list(all_extensions)}. "
            "You MUST write a custom Python script and execute it via 'run_python' to analyze "
            "the data split layout (e.g., loading a CSV/Parquet file using pandas to check class distributions, "
            "or parsing a metadata JSON)."
        )
    }, indent=2)

@tool
def inspect_annotation_format() -> str:
    """
    Examine a sample annotation file by parsing the pre-cached workspace structure dictionary.
    Locates annotation extensions (.xml, .json, .txt, .csv) across standard CV or NLP structures,
    then reads a 300-character snippet.
    
    Returns a JSON string containing file info, format guess, or a descriptive error 
    for the agent to craft custom Python code if it fails.
    """
    ds = workspace_mgr.workspace
    structure = workspace_mgr.structure_cache

    if not ds:
        return json.dumps({"error": "No workspace set. Run 'copy_dataset' first."})
    if not structure:
        return json.dumps({"error": "Directory structure cache is missing. Run 'explore_structure' first."})

    target_rel_path = None
    target_ext = None
    
    # Supported annotation formats in order of structural specificity
    ANNOTATION_EXTS = [".xml", ".json", ".txt", ".csv"]

    # -----------------------------------------------------------------
    # Step 1: Scan Cached Structure to Find an Annotation File
    # -----------------------------------------------------------------
    
    # Priority A: Check for explicit "labels" or "annotations" folders (Standard CV)
    for rel_path, info in structure.items():
        path_lower = rel_path.lower()
        if "label" in path_lower or "annot" in path_lower:
            ext_counts = info.get("extensions", {})
            for ext in ANNOTATION_EXTS:
                if ext in ext_counts and ext_counts[ext] > 0:
                    target_rel_path = rel_path
                    target_ext = ext
                    break
        if target_rel_path:
            break

    # Priority B: Fallback to any train/val/test split folder holding annotation extensions
    if not target_rel_path:
        for rel_path, info in structure.items():
            path_lower = rel_path.lower()
            if any(s in path_lower for s in ["train", "val", "test"]):
                ext_counts = info.get("extensions", {})
                for ext in ANNOTATION_EXTS:
                    if ext in ext_counts and ext_counts[ext] > 0:
                        target_rel_path = rel_path
                        target_ext = ext
                        break
            if target_rel_path:
                break

    # Priority C: Fallback to Root directory (Standard NLP / flat tabular datasets)
    if not target_rel_path and "root" in structure:
        ext_counts = structure["root"].get("extensions", {})
        for ext in ANNOTATION_EXTS:
            if ext in ext_counts and ext_counts[ext] > 0:
                target_rel_path = "root"
                target_ext = ext
                break

    # -----------------------------------------------------------------
    # Step 2: Extract Snippet or Return Structural Guidance
    # -----------------------------------------------------------------
    if not target_rel_path:
        # Fallback payload detailing exactly what went wrong so the agent can generate custom code
        all_found_extensions = set()
        for info in structure.values():
            all_found_extensions.update(info.get("extensions", {}).keys())

        return json.dumps({
            "status": "failed",
            "reason": "Could not automatically locate standard annotation files (.xml, .json, .txt, .csv) in expected label/split directories.",
            "instruction_for_agent": (
                "The target dataset has an unconventional structure or format. Review the output of 'explore_structure'. "
                f"The environment contains these extensions: {list(all_found_extensions)}. "
                "Use the 'run_python' tool to manually inspect the file contents or map custom paths "
                "tailored to this specific schema."
            )
        }, indent=2)

    # Convert "root" tag back to actual workspace base path
    actual_dir = ds if target_rel_path == "root" else ds / target_rel_path

    # Grab the first file matching the discovered target extension
    try:
        sample_file = next(actual_dir.glob(f"*{target_ext}"))
    except StopIteration:
        return json.dumps({"error": f"Cache mismatch. Expected file with {target_ext} in {target_rel_path} but found none."})

    # Read the file snippet safely
    snippet = ""
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            snippet = f.read(300)
    except Exception as e:
        snippet = f"[Unreadable/Binary File Error: {str(e)}]"

    # Heuristic format guesser
    format_guess = "unknown"
    if target_ext == ".xml" and ("<annotation>" in snippet.lower() or "<object>" in snippet.lower()):
        format_guess = "Pascal VOC XML"
    elif target_ext == ".json":
        format_guess = "JSON (COCO, custom metadata, or NLP intents)"
    elif target_ext == ".txt" and len(snippet.strip().split("\n")[0].split()) == 5:
        format_guess = "YOLO (txt bounding boxes)"
    elif target_ext == ".csv":
        format_guess = "CSV/TSV Tabular Dataset"

    return json.dumps({
        "status": "success",
        "inspected_directory": target_rel_path,
        "sample_file": str(sample_file.relative_to(ds)),
        "extension": target_ext,
        "format_guess": format_guess,
        "snippet": snippet
    }, indent=2)

@tool
def run_python(code: str) -> str:
    """
    Execute Python code in a sandboxed subprocess with the workspace as current directory.
    Returns stdout, stderr, and any generated plot files.
    """
    ws = workspace_mgr.workspace
    if not ws:
        return "Error: No workspace set. Use copy_dataset first."
    script_path = ws / "_temp_script.py"
    script_path.write_text(code)
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True, timeout=60, cwd=str(ws)
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        images = list(ws.glob("*.png")) + list(ws.glob("*.jpg")) + list(ws.glob("*.svg"))
        if images:
            output += "\n[Plot files: " + ", ".join(img.name for img in images) + "]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (60s)."
    finally:
        if script_path.exists():
            script_path.unlink()

@tool
def ask_user(question: str) -> str:
    """
    Ask the user a question and return their response.
    Use this for path re-entry, confirmation, or clarification.
    """
    return input(f"\n[Agent]: {question}\n> ")