#!/usr/bin/env python3
"""
Batch construct prompts for PR-based coding assessments.

Reads repository information from gpt_prompt_info_statistics.json and constructs
complete prompts by filling in placeholders with actual data.
"""

import argparse
import os
import json
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


PROMPT_TEMPLATE = """You are an experienced programming instructor. Convert a single GitHub Pull Request (PR) into a **realistic, agent-style coding assessment**.

**Only output the question** between the required delimiters. Do **not** include any solutions, assets, metadata, or diffs.

## Inputs (provided below via placeholders)

* `{PR_JSON}`: Raw PR object (title, body, comments, commits, timestamps, etc.). + {PR_JSON_ACTUAL}

* `{PATCH_UNIFIED_DIFF}`: Full unified diff of the PR. **For reference only. Do not reveal any of it.** + {PATCH_UNIFIED_DIFF_ACTUAL}

* `{PATCH_METADATA}`: Files changed + per-file additions/deletions. + {PATCH_METADATA_ACTUAL}

* `{BASE_COMMIT_SHA}`: The base commit SHA the PR applies to. + {BASE_COMMIT_SHA_ACTUAL}

* `{REPO_METADATA}`: Repo identifiers (owner/name/URLs). + {REPO_METADATA_ACTUAL}

* `{BASE_CODE_TAR_PATH}` or `{BASE_CODE_DIR}`: Base commit code bundle/path. + {BASE_CODE_PATH_ACTUAL}

* `{ORACLE_FILES}` (optional): Original versions of the edited files at `{BASE_COMMIT_SHA}` for quoting **small context snippets** only. + {ORACLE_FILES_ACTUAL}

* `{PROBLEM_STATEMENT}` (optional): If present in the PR; may be empty. + {PROBLEM_STATEMENT_ACTUAL}

* `{TEST_PATCH_UNIFIED_DIFF}` (optional): Whether tests were changed (reference only; **do not reveal code**). + {TEST_PATCH_ACTUAL}

* `{AVG_FILES}`, `{TOTAL_FILES}`, `{TOTAL_LINES}` (optional): Aggregate repo stats. + {REPO_STATS_ACTUAL}

## Hard Rules

1. **No Leakage**: Do **not** quote, paraphrase, summarize, or expose any content from `{PATCH_UNIFIED_DIFF}` or `{TEST_PATCH_UNIFIED_DIFF}`.

2. **Grounding**: All requirements must be **strictly derived** from `{PR_JSON}` + code context at `{BASE_COMMIT_SHA}`. No hallucinated APIs/files/behaviors.

3. **Edit Scope**: The "Files to modify" list in the question must be a **subset of** `{PATCH_METADATA.changed_files}` and should reflect the actual loci of change.

4. **Context Snippets**: If helpful, you may include **small excerpts** from `{ORACLE_FILES}` to illustrate the **pre-change** context only (avoid large blocks). For large references, use link stubs:

   ```

   <github_code_start>https://github.com/{owner}/{repo}/blob/{BASE_COMMIT_SHA}/path/to/file.py#L100-L160</github_code_start>

   ```

5. **Verifiability**: Phrase acceptance criteria so that a grader could verify by applying a correct patch to `{BASE_COMMIT_SHA}` and running the project. Do **not** reveal expected code.

6. **Unsuitable PRs**: If inputs are insufficient (e.g., doc-only/binary-only changes, missing base code), output **only** `BAD_PR` inside the Question block.

## Output Format (produce **only** this block)

````

<|Question Begin|>

Title: {Concise, domain-relevant title}

Scenario (agent-style, production context):

- Briefly describe the real-world motivation and the subsystem involved, derived from {PR_JSON}.

- Keep it practical (e.g., configuration loading, IO pipeline, API contract fix).

Available Materials:

- Base commit: {BASE_COMMIT_SHA}

- Code bundle: {BASE_CODE_TAR_PATH} or {BASE_CODE_DIR}

- Repository: {REPO_METADATA.owner}/{REPO_METADATA.name}

- (Optional) Context excerpts (pre-change) from ORACLE_FILES:

  ```python

  # file: path/to/file.py @ {BASE_COMMIT_SHA}

  # short snippet showing current API/signature/usage relevant to the change

````

Or link stubs:

<github_code_start>[https://github.com/{owner}/{repo}/blob/{BASE_COMMIT_SHA}/path/to/file.py#L120-L168](https://github.com/{owner}/{repo}/blob/{BASE_COMMIT_SHA}/path/to/file.py#L120-L168)</github_code_start>

Your Task:

* Implement the required fix/feature by editing **only** these files:

  * path/to/fileA.py

  * path/to/fileB.json

    (All listed files must be a subset of {PATCH_METADATA.changed_files}.)

* Preserve public contracts unless the change is explicitly required by the scenario.

Functional Requirements:

* Bullet the concrete behaviors expected after the change (input/output, config keys, error handling, edge cases), reconstructed from {PR_JSON}.

* Avoid spoilers (no code, no diff lines, no exact identifiers beyond what exists at base).

Constraints:

* Performance/complexity expectations if implied by the PR (e.g., do not introduce O(N^2) in hot path).

* Backward compatibility or migration notes as applicable.

* Do not modify files outside the scope above.

How We Evaluate:

* Apply the candidate's patch to {BASE_COMMIT_SHA}.

* Run the minimal command(s) to exercise the change (e.g., `python main.py --config path/to/config.json`) and/or project tests if they exist.

* Success signals: specify observable outcomes (logs, CLI output, behavior) without revealing implementation.

Submission Format:

* Submit a unified diff patch that applies cleanly to {BASE_COMMIT_SHA}.

* Do not include compiled artifacts or large assets.

(If the PR is unsuitable or inputs insufficient, output BAD_PR.)

<|Question End|>

```

## Final Instruction

- Output **only** the `<|Question Begin|>…<|Question End|>` block.  

- Do **not** include any additional sections, notes, or explanations.
"""


def format_pr_json(pr_data: Dict) -> str:
    """Format PR JSON data for insertion."""
    if not pr_data:
        return "PR JSON not available"
    
    # Extract key fields
    formatted = []
    formatted.append(f"PR #{pr_data.get('number', 'N/A')}: {pr_data.get('title', 'N/A')}")
    formatted.append(f"State: {pr_data.get('state', 'N/A')}")
    formatted.append(f"Created: {pr_data.get('created_at', 'N/A')}")
    formatted.append(f"Merged: {pr_data.get('merged_at', 'N/A') if pr_data.get('merged_at') else 'Not merged'}")
    formatted.append(f"Author: {pr_data.get('user', 'N/A')}")
    formatted.append(f"URL: {pr_data.get('html_url', 'N/A')}")
    formatted.append(f"\nBody:\n{pr_data.get('body', 'No description')[:500]}{'...' if len(pr_data.get('body', '')) > 500 else ''}")
    formatted.append(f"\nChanges: +{pr_data.get('additions', 0)}/-{pr_data.get('deletions', 0)} lines in {pr_data.get('changed_files', 0)} files")
    
    return "\n".join(formatted)


def format_patch_diff(patch_content: Optional[str], max_length: int = 10000) -> str:
    """Format patch diff for insertion."""
    if not patch_content:
        return "Patch not available"
    
    if len(patch_content) > max_length:
        return f"Patch available (truncated, total length: {len(patch_content)} chars):\n{patch_content[:max_length]}...\n[Full patch available in tar file]"
    
    return f"Patch ({len(patch_content)} chars):\n{patch_content}"


def format_patch_metadata(patch_stats: Optional[Dict], changed_files: Optional[List]) -> str:
    """Format patch metadata for insertion."""
    if not patch_stats and not changed_files:
        return "Patch metadata not available"
    
    formatted = []
    if patch_stats:
        formatted.append(f"Files changed: {patch_stats.get('files_changed', 0)}")
        formatted.append(f"Lines added: {patch_stats.get('lines_added', 0)}")
        formatted.append(f"Lines deleted: {patch_stats.get('lines_deleted', 0)}")
        formatted.append(f"Net change: {patch_stats.get('net_change', 0)}")
    
    if changed_files:
        formatted.append(f"\nChanged files ({len(changed_files)}):")
        for file_info in changed_files[:20]:  # Limit to first 20
            if isinstance(file_info, dict):
                formatted.append(f"  - {file_info.get('file', 'Unknown')}: +{file_info.get('added', 0)}/-{file_info.get('deleted', 0)}")
            else:
                formatted.append(f"  - {file_info}")
        if len(changed_files) > 20:
            formatted.append(f"  ... and {len(changed_files) - 20} more files")
    
    return "\n".join(formatted)


def format_repo_metadata(repo_data: Dict) -> str:
    """Format repository metadata for insertion."""
    if not repo_data:
        return "Repository metadata not available"
    
    formatted = []
    formatted.append(f"Full name: {repo_data.get('full_name', 'Unknown/Unknown')}")
    formatted.append(f"Owner: {repo_data.get('owner', 'Unknown')}")
    formatted.append(f"Name: {repo_data.get('name', 'Unknown')}")
    formatted.append(f"URL: {repo_data.get('html_url', 'N/A')}")
    
    return "\n".join(formatted)


def format_base_code_path(local_paths: Dict, base_commit: str, repo_data: Dict) -> str:
    """Format base code path/URL for insertion."""
    if not local_paths:
        return "Base code path not available"
    
    # Only use source code directory (not tar file, which contains PR metadata, not code)
    code_dir = local_paths.get('source_code_dir')
    
    formatted = []
    if code_dir and os.path.exists(code_dir):
        formatted.append(f"Base code directory: {code_dir}")
        
        # Count files in directory
        try:
            file_count = sum(1 for root, dirs, files in os.walk(code_dir) for f in files)
            formatted.append(f"Total files in directory: {file_count}")
        except:
            pass
    
    # GitHub link to base commit tree
    owner = repo_data.get('owner', 'Unknown') if repo_data else 'Unknown'
    repo_name = repo_data.get('name', 'Unknown') if repo_data else 'Unknown'
    formatted.append(f"GitHub tree (base commit): https://github.com/{owner}/{repo_name}/tree/{base_commit}")
    
    return "\n".join(formatted)


def format_oracle_files(oracle_files: Optional[List], edit_files: Optional[List], 
                       base_commit: str, repo_data: Dict) -> str:
    """Format oracle files for insertion."""
    if not oracle_files:
        if edit_files:
            return f"Oracle files: {len(edit_files)} files were modified (see edit_files list). Full content available in base commit code bundle."
        return "Oracle files: Not available (empty or not present in SWE-Bench data)"
    
    formatted = []
    formatted.append(f"Oracle files count: {len(oracle_files)}")
    formatted.append("\nOracle files (showing first 3 file contents, truncated):")
    
    for i, oracle_content in enumerate(oracle_files[:3], 1):
        if isinstance(oracle_content, str):
            # Try to extract filename from edit_files if available
            filename = edit_files[i-1] if edit_files and i-1 < len(edit_files) else f"file_{i}"
            content_preview = oracle_content[:500] + "..." if len(oracle_content) > 500 else oracle_content
            formatted.append(f"\n--- {filename} ---")
            formatted.append(content_preview)
    
    if len(oracle_files) > 3:
        formatted.append(f"\n... and {len(oracle_files) - 3} more oracle files")
    
    # Add GitHub links for all edit files
    if edit_files:
        owner = repo_data.get('owner', 'Unknown') if repo_data else 'Unknown'
        repo_name = repo_data.get('name', 'Unknown') if repo_data else 'Unknown'
        formatted.append(f"\nAll edited files in base commit:")
        for edit_file in edit_files[:10]:
            formatted.append(f"  https://github.com/{owner}/{repo_name}/blob/{base_commit}/{edit_file}")
        if len(edit_files) > 10:
            formatted.append(f"  ... and {len(edit_files) - 10} more files")
    
    return "\n".join(formatted)


def format_problem_statement(problem_statement: Optional[Dict]) -> str:
    """Format problem statement for insertion."""
    if not problem_statement or problem_statement == {}:
        return "Problem statement: Not available (empty in SWE-Bench data)"
    
    formatted = []
    if isinstance(problem_statement, dict):
        if problem_statement.get('title'):
            formatted.append(f"Title: {problem_statement.get('title')}")
        if problem_statement.get('body'):
            body = problem_statement.get('body', '')
            formatted.append(f"\nBody:\n{body[:1000]}{'...' if len(body) > 1000 else ''}")
    else:
        formatted.append(str(problem_statement)[:500])
    
    return "\n".join(formatted) if formatted else "Problem statement: Available but format unknown"


def format_test_patch(test_patch: Optional[str]) -> str:
    """Format test patch for insertion."""
    if not test_patch or test_patch.strip() == "":
        return "Test patch: Not available (no tests were changed or test patch is empty)"
    
    return f"Test patch available (length: {len(test_patch)} chars). For reference only, do not reveal test code."


def format_repo_stats(code_stats: Optional[Dict]) -> str:
    """Format repository statistics for insertion."""
    if not code_stats:
        return "Repository stats: Not available"
    
    formatted = []
    formatted.append(f"Total files: {code_stats.get('total_files', 0)}")
    formatted.append(f"Total lines: {code_stats.get('total_lines', 0):,}")
    formatted.append(f"Python files: {code_stats.get('python_files', 0)}")
    formatted.append(f"Python lines: {code_stats.get('python_lines', 0):,}")
    
    return "\n".join(formatted)


def construct_prompt(repo_info: Dict, prompt_template: str) -> str:
    """Construct a complete prompt from repository information."""
    
    # Extract data
    pr_data = repo_info.get('pr', {})
    patch_stats = repo_info.get('patch', {})
    swebench_data = repo_info.get('swebench', {})
    commit_data = repo_info.get('commit', {})
    repo_data = repo_info.get('repository', {})
    local_paths = repo_info.get('local_paths', {})
    code_stats = repo_info.get('code_statistics', {})
    
    base_commit = swebench_data.get('base_commit') or commit_data.get('sha', 'N/A')
    
    # Format each section
    pr_json_actual = format_pr_json(pr_data)
    
    # Read patch content from tar if available
    patch_content = None
    
    # First try to get from swebench data (may contain full patch)
    patch_content = swebench_data.get('patch', '') or swebench_data.get('patch_preview', '')
    
    # If not available, read from tar file
    if not patch_content and local_paths.get('tar_file') and os.path.exists(local_paths['tar_file']):
        try:
            import tarfile
            with tarfile.open(local_paths['tar_file'], 'r') as tar:
                for member in tar.getmembers():
                    filename = member.name.lstrip('./').replace('\\', '/')
                    if filename.endswith('.patch'):
                        patch_data = tar.extractfile(member).read()
                        patch_content = patch_data.decode('utf-8', errors='ignore')
                        break
        except Exception as e:
            print(f"Warning: Could not read patch from tar: {str(e)}")
    
    patch_diff_actual = format_patch_diff(patch_content)
    patch_metadata_actual = format_patch_metadata(patch_stats, patch_stats.get('changed_files') if patch_stats else None)
    
    base_commit_actual = base_commit
    repo_metadata_actual = format_repo_metadata(repo_data)
    base_code_path_actual = format_base_code_path(local_paths, base_commit, repo_data)
    
    oracle_files = swebench_data.get('oracle_files', [])
    edit_files = swebench_data.get('edit_files', [])
    oracle_files_actual = format_oracle_files(oracle_files, edit_files, base_commit, repo_data)
    
    problem_statement_actual = format_problem_statement(swebench_data.get('problem_statement'))
    
    test_patch_content = swebench_data.get('test_patch', '')
    test_patch_actual = format_test_patch(test_patch_content)
    
    repo_stats_actual = format_repo_stats(code_stats)
    
    # Replace placeholders
    prompt = prompt_template
    prompt = prompt.replace('{PR_JSON_ACTUAL}', pr_json_actual)
    prompt = prompt.replace('{PATCH_UNIFIED_DIFF_ACTUAL}', patch_diff_actual)
    prompt = prompt.replace('{PATCH_METADATA_ACTUAL}', patch_metadata_actual)
    prompt = prompt.replace('{BASE_COMMIT_SHA_ACTUAL}', base_commit_actual)
    prompt = prompt.replace('{REPO_METADATA_ACTUAL}', repo_metadata_actual)
    prompt = prompt.replace('{BASE_CODE_PATH_ACTUAL}', base_code_path_actual)
    prompt = prompt.replace('{ORACLE_FILES_ACTUAL}', oracle_files_actual)
    prompt = prompt.replace('{PROBLEM_STATEMENT_ACTUAL}', problem_statement_actual)
    prompt = prompt.replace('{TEST_PATCH_ACTUAL}', test_patch_actual)
    prompt = prompt.replace('{REPO_STATS_ACTUAL}', repo_stats_actual)
    
    # Replace owner/repo placeholders in template
    owner = repo_data.get('owner', 'Unknown') if repo_data else 'Unknown'
    repo_name = repo_data.get('name', 'Unknown') if repo_data else 'Unknown'
    prompt = prompt.replace('{owner}', owner)
    prompt = prompt.replace('{repo}', repo_name)
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description='Batch construct prompts for PR-based coding assessments'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default='/home/xubing/code/MMDataKit/xubing_dataprocess/sd_swe/gpt_prompt_info_statistics.json',
        help='Input JSON file with repository information'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/xubing/code/MMDataKit/xubing_dataprocess/sd_swe/prompts',
        help='Output directory for constructed prompts'
    )
    parser.add_argument(
        '--output_jsonl',
        type=str,
        default=None,
        help='Optional: Output JSONL file with all prompts (one per line)'
    )
    parser.add_argument(
        '--max_repos',
        type=int,
        default=None,
        help='Maximum number of repositories to process (for testing)'
    )
    parser.add_argument(
        '--test_sample',
        type=int,
        default=None,
        help='If specified, only process first N repositories and show preview (overrides --max_repos)'
    )
    parser.add_argument(
        '--filter_successful',
        action='store_true',
        help='Only process successful repositories'
    )
    
    args = parser.parse_args()
    
    # Read input JSON
    print(f"Reading repository information from: {args.input_json}")
    if not os.path.exists(args.input_json):
        print(f"Error: Input file not found: {args.input_json}")
        return
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    repositories = data.get('repositories', [])
    print(f"Found {len(repositories)} repositories")
    
    # Filter if needed
    if args.filter_successful:
        repositories = [r for r in repositories if r.get('success', False)]
        print(f"After filtering successful: {len(repositories)} repositories")
    
    # Limit if specified (test_sample overrides max_repos)
    if args.test_sample is not None:
        repositories = repositories[:args.test_sample]
        print(f"Test mode: Processing {len(repositories)} repository(ies)")
    elif args.max_repos:
        repositories = repositories[:args.max_repos]
        print(f"Limited to {len(repositories)} repositories")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each repository
    constructed_prompts = []
    
    for repo_info in tqdm(repositories, desc="Constructing prompts"):
        repo_name = repo_info.get('repo_name', 'Unknown')
        tar_file_name = repo_info.get('tar_file_name', repo_name)
        
        try:
            prompt = construct_prompt(repo_info, PROMPT_TEMPLATE)
            
            # Save individual prompt file
            output_file = os.path.join(args.output_dir, f"{tar_file_name}.prompt.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            constructed_prompts.append({
                'repo_name': repo_name,
                'tar_file_name': tar_file_name,
                'prompt': prompt,
                'output_file': output_file
            })
            
        except Exception as e:
            print(f"\nError constructing prompt for {repo_name}: {str(e)}")
            continue
    
    # Save JSONL if requested
    if args.output_jsonl:
        print(f"\nSaving all prompts to JSONL: {args.output_jsonl}")
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for item in constructed_prompts:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    # Print summary
    print(f"\n{'='*80}")
    print("CONSTRUCTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total prompts constructed: {len(constructed_prompts)}")
    print(f"Output directory: {args.output_dir}")
    
    if args.test_sample and constructed_prompts:
        print(f"\n{'='*80}")
        print("SAMPLE PROMPT PREVIEW")
        print(f"{'='*80}")
        sample = constructed_prompts[0]
        print(f"Repository: {sample['repo_name']}")
        print(f"Output file: {sample['output_file']}")
        print(f"\nPrompt preview (first 1000 chars):")
        print("-" * 80)
        print(sample['prompt'][:1000])
        print("...")


if __name__ == '__main__':
    main()

