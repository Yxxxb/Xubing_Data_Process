import argparse
import os
import json
import tarfile
from typing import List, Dict, Optional
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


def count_code_statistics(source_code_dir: str) -> Dict:
    """Count code statistics from source code directory.
    
    Args:
        source_code_dir (str): Path to source code directory.
        
    Returns:
        Dict: Statistics including file counts, line counts, etc.
    """
    stats = {
        'total_files': 0,
        'total_lines': 0,
        'total_bytes': 0,
        'python_files': 0,
        'python_lines': 0,
        'file_extensions': defaultdict(int),
        'largest_files': []
    }
    
    if not os.path.exists(source_code_dir):
        return stats
    
    largest_files = []
    
    for root, dirs, files in os.walk(source_code_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            try:
                file_size = os.path.getsize(file_path)
                stats['total_bytes'] += file_size
                stats['total_files'] += 1
                
                # Count by extension
                _, ext = os.path.splitext(file)
                stats['file_extensions'][ext] += 1
                
                # Count lines
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        stats['total_lines'] += lines
                        
                        if ext == '.py':
                            stats['python_files'] += 1
                            stats['python_lines'] += lines
                        
                        # Track largest files
                        largest_files.append({
                            'path': os.path.relpath(file_path, source_code_dir),
                            'size': file_size,
                            'lines': lines,
                            'extension': ext
                        })
                except:
                    pass
                    
            except Exception:
                pass
    
    # Sort and get top 10 largest files
    largest_files.sort(key=lambda x: x['lines'], reverse=True)
    stats['largest_files'] = largest_files[:10]
    stats['file_extensions'] = dict(stats['file_extensions'])
    
    return stats


def get_pr_info_from_tar(tar_path: str, repo_name: str) -> Optional[Dict]:
    """Extract PR information from tar file.
    
    Args:
        tar_path (str): Path to tar file.
        repo_name (str): Repository name to match.
        
    Returns:
        Optional[Dict]: PR information or None.
    """
    try:
        pr_json = None
        swebench_json = None
        patch_content = None
        issuecomments = []
        
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                filename = member.name
                # Normalize filename (remove leading ./ and handle path separators)
                normalized_name = filename.lstrip('./').replace('\\', '/')
                
                # Check issue comments first (most specific condition)
                if '#issuecomment-' in normalized_name and normalized_name.endswith('.json'):
                    try:
                        json_data = tar.extractfile(member).read()
                        comment_data = json.loads(json_data)
                        # Store in same format as filter_swe_sample_40k_test_single.py
                        issuecomments.append({
                            'filename': normalized_name,
                            'data': comment_data
                        })
                    except Exception as e:
                        print(f"Warning: Failed to parse issue comment {filename}: {str(e)}")
                        pass
                
                # Then check for PR/SWE-Bench JSON files
                elif normalized_name.endswith('.json') and '#pull#' in normalized_name:
                    try:
                        json_data = tar.extractfile(member).read()
                        if normalized_name.endswith('_swebench.json'):
                            swebench_json = json.loads(json_data)
                        else:
                            # Regular PR JSON (not swebench, not issuecomment)
                            pr_json = json.loads(json_data)
                    except Exception as e:
                        print(f"Warning: Failed to parse PR/SWE-Bench JSON {filename}: {str(e)}")
                        pass
                
                # Then check for patch files
                elif normalized_name.endswith('.patch'):
                    try:
                        patch_data = tar.extractfile(member).read()
                        patch_content = patch_data.decode('utf-8', errors='ignore')
                    except Exception as e:
                        print(f"Warning: Failed to parse patch {filename}: {str(e)}")
                        pass
        
        return {
            'pr_json': pr_json,
            'swebench_json': swebench_json,
            'patch': patch_content,
            'issuecomments': issuecomments
        }
        
    except Exception as e:
        print(f"Error reading tar file {tar_path}: {str(e)}")
        return None


def analyze_patch(patch_content: Optional[str]) -> Dict:
    """Analyze patch content to extract change statistics.
    
    Args:
        patch_content (str): Patch content.
        
    Returns:
        Dict: Patch statistics.
    """
    stats = {
        'patch_length': len(patch_content) if patch_content else 0,
        'files_changed': 0,
        'lines_added': 0,
        'lines_deleted': 0,
        'changed_files': []
    }
    
    if not patch_content:
        return stats
    
    # Count files changed (lines starting with "diff --git")
    files = []
    current_file = None
    lines_added = 0
    lines_deleted = 0
    
    for line in patch_content.split('\n'):
        if line.startswith('diff --git'):
            if current_file:
                files.append({
                    'file': current_file,
                    'added': lines_added,
                    'deleted': lines_deleted
                })
            # Extract filename from "diff --git a/path b/path"
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3].replace('b/', '', 1)
            lines_added = 0
            lines_deleted = 0
        elif line.startswith('+') and not line.startswith('+++'):
            lines_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_deleted += 1
    
    if current_file:
        files.append({
            'file': current_file,
            'added': lines_added,
            'deleted': lines_deleted
        })
    
    stats['files_changed'] = len(files)
    stats['lines_added'] = sum(f['added'] for f in files)
    stats['lines_deleted'] = sum(f['deleted'] for f in files)
    stats['changed_files'] = files[:20]  # Limit to first 20 files
    
    return stats


def collect_repository_info(commit_dir: str, tar_base_path: str) -> Optional[Dict]:
    """Collect comprehensive information about a repository.
    
    Args:
        commit_dir (str): Directory containing commit code.
        tar_base_path (str): Base path to tar files.
        
    Returns:
        Optional[Dict]: Repository information.
    """
    info = {
        'tar_file_name': None,
        'repo_name': None,
        'success': False,
        'error': None
    }
    
    try:
        # Extract repo name from directory name (format: repo#owner#pull#number)
        dir_name = os.path.basename(commit_dir)
        info['tar_file_name'] = dir_name
        info['repo_name'] = dir_name
        
        # Read commit_info.json
        commit_info_file = os.path.join(commit_dir, 'commit_info.json')
        if not os.path.exists(commit_info_file):
            info['error'] = "commit_info.json not found"
            return info
        
        with open(commit_info_file, 'r', encoding='utf-8') as f:
            commit_info = json.load(f)
        
        # Extract commit information
        commit_data = commit_info.get('commit', {})
        info['commit'] = {
            'sha': commit_info.get('sha'),
            'message': commit_data.get('message', ''),
            'author': {
                'name': commit_data.get('author', {}).get('name'),
                'email': commit_data.get('author', {}).get('email'),
                'date': commit_data.get('author', {}).get('date')
            },
            'html_url': commit_info.get('html_url'),
            'api_url': commit_info.get('url')
        }
        
        # Extract repository information
        repo_data = commit_info.get('repository') or commit_info.get('repo', {})
        repo_name = repo_data.get('full_name') or repo_data.get('name', 'Unknown/Unknown')
        info['repository'] = {
            'full_name': repo_name,
            'name': repo_name.split('/')[-1] if '/' in repo_name else repo_name,
            'owner': repo_name.split('/')[0] if '/' in repo_name else 'Unknown',
            'html_url': f"https://github.com/{repo_name}",
            'api_url': repo_data.get('url') or commit_info.get('repository', {}).get('url')
        }
        
        # Find and read tar file
        tar_file_name = f"{dir_name}.tar"
        tar_path = os.path.join(tar_base_path, tar_file_name)
        
        if not os.path.exists(tar_path):
            info['error'] = f"Tar file not found: {tar_file_name}"
            return info
        
        info['local_paths'] = {
            'commit_code_dir': commit_dir,
            'tar_file': tar_path,
            'source_code_dir': os.path.join(commit_dir, 'source_code')
        }
        
        # Read PR and SWE-Bench info from tar
        pr_info = get_pr_info_from_tar(tar_path, dir_name)
        
        if pr_info:
            # PR JSON information
            if pr_info.get('pr_json'):
                pr_data = pr_info['pr_json']
                info['pr'] = {
                    'number': pr_data.get('number'),
                    'title': pr_data.get('title', ''),
                    'body': pr_data.get('body', ''),
                    'state': pr_data.get('state'),
                    'created_at': pr_data.get('created_at'),
                    'updated_at': pr_data.get('updated_at'),
                    'merged_at': pr_data.get('merged_at'),
                    'html_url': pr_data.get('html_url'),
                    'user': pr_data.get('user', {}).get('login') if pr_data.get('user') else None,
                    'additions': pr_data.get('additions'),
                    'deletions': pr_data.get('deletions'),
                    'changed_files': pr_data.get('changed_files'),
                    'commits': pr_data.get('commits')
                }
            
            # SWE-Bench JSON information
            if pr_info.get('swebench_json'):
                swebench_data = pr_info['swebench_json']
                info['swebench'] = {
                    'instance_id': swebench_data.get('instance_id'),
                    'repo': swebench_data.get('repo'),
                    'base_commit': swebench_data.get('base_commit'),
                    'problem_statement': swebench_data.get('problem_statement', {}),
                    'edit_files': swebench_data.get('edit_files', []),
                    'oracle_files': swebench_data.get('oracle_files', []),
                    'test_patch': swebench_data.get('test_patch', ''),
                    'patch_preview': swebench_data.get('patch', '')[:500] if swebench_data.get('patch') else ''
                }
            
            # Patch information
            if pr_info.get('patch'):
                patch_stats = analyze_patch(pr_info['patch'])
                info['patch'] = {
                    'length': patch_stats['patch_length'],
                    'files_changed': patch_stats['files_changed'],
                    'lines_added': patch_stats['lines_added'],
                    'lines_deleted': patch_stats['lines_deleted'],
                    'net_change': patch_stats['lines_added'] - patch_stats['lines_deleted'],
                    'changed_files': patch_stats['changed_files']
                }
            
            # Issue comments
            issuecomments_list = pr_info.get('issuecomments', [])
            if issuecomments_list:
                info['issue_comments'] = []
                info['issue_comments_count'] = len(issuecomments_list)
                for comment in issuecomments_list:
                    # Handle both dict format (from get_pr_info_from_tar) and direct JSON format
                    comment_data = comment.get('data') if isinstance(comment, dict) and 'data' in comment else comment
                    if comment_data:
                        info['issue_comments'].append({
                            'id': comment_data.get('id'),
                            'body': comment_data.get('body', ''),
                            'user': comment_data.get('user', {}).get('login') if comment_data.get('user') else None,
                            'created_at': comment_data.get('created_at'),
                            'html_url': comment_data.get('html_url')
                        })
            else:
                info['issue_comments'] = []
                info['issue_comments_count'] = 0
        
        # Count code statistics
        source_code_dir = info['local_paths']['source_code_dir']
        if not os.path.exists(source_code_dir):
            info['error'] = "source_code directory not found (repository was not downloaded or skipped)"
            info['success'] = False
            return info
        
        # Only process if source_code directory exists (repository was actually downloaded)
        code_stats = count_code_statistics(source_code_dir)
        info['code_statistics'] = code_stats
        
        # Calculate difficulty metrics
        if code_stats['total_files'] > 0:
            if info.get('patch'):
                patch_file_count = info['patch']['files_changed']
                info['difficulty_metrics'] = {
                    'total_files_in_repo': code_stats['total_files'],
                    'files_changed_count': patch_file_count,
                    'files_changed_ratio': patch_file_count / code_stats['total_files'],
                    'total_lines_in_repo': code_stats['total_lines'],
                    'lines_added': info['patch'].get('lines_added', 0),
                    'lines_deleted': info['patch'].get('lines_deleted', 0),
                    'net_lines_changed': info['patch'].get('net_change', 0),
                    'lines_changed_ratio': (info['patch'].get('lines_added', 0) + info['patch'].get('lines_deleted', 0)) / max(code_stats['total_lines'], 1),
                    'pr_body_length': len(info.get('pr', {}).get('body', '')),
                    'commit_message_length': len(info.get('commit', {}).get('message', '')),
                    'python_file_count': code_stats['python_files'],
                    'python_line_count': code_stats['python_lines']
                }
        
        info['success'] = True
        
    except Exception as e:
        info['error'] = str(e)
    
    return info


def main():
    parser = argparse.ArgumentParser(description='Collect repository information for GPT prompt construction')
    parser.add_argument('--commit_raw_code_dir', type=str,
                        default='/data/xubing/oss_swe/swebench_sample_40k_commit_raw_code',
                        help='Directory containing downloaded commit code')
    parser.add_argument('--tar_base_path', type=str,
                        default='/data/xubing/oss_swe/swebench_sample_40k',
                        help='Base path to tar files')
    parser.add_argument('--output', type=str,
                        default='/home/xubing/code/MMDataKit/xubing_dataprocess/sd_swe/gpt_prompt_info_statistics.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    print(f"Scanning commit directories in: {args.commit_raw_code_dir}")
    
    # Find all commit directories
    commit_dirs = []
    if os.path.exists(args.commit_raw_code_dir):
        for item in os.listdir(args.commit_raw_code_dir):
            item_path = os.path.join(args.commit_raw_code_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'commit_info.json')):
                commit_dirs.append(item_path)
    
    print(f"Found {len(commit_dirs)} commit directories")
    
    if not commit_dirs:
        print("No commit directories found!")
        return
    
    # Collect information for each repository
    all_info = []
    skipped_info = []
    
    for commit_dir in tqdm(commit_dirs, desc="Collecting repository info"):
        info = collect_repository_info(commit_dir, args.tar_base_path)
        if info:
            # Only include repositories that were actually downloaded (have source_code directory)
            source_code_dir = os.path.join(commit_dir, 'source_code')
            if os.path.exists(source_code_dir):
                all_info.append(info)
            else:
                skipped_info.append({
                    'repo_name': info.get('repo_name', 'Unknown'),
                    'reason': 'source_code directory not found (not downloaded)'
                })
    
    # Sort by repo name
    all_info.sort(key=lambda x: x.get('repo_name', ''))
    
    # Create summary statistics
    summary = {
        'total_repositories': len(all_info),
        'total_skipped': len(skipped_info),
        'successful': sum(1 for x in all_info if x.get('success')),
        'failed': sum(1 for x in all_info if not x.get('success')),
        'statistics': {
            'avg_files_per_repo': 0,
            'avg_lines_per_repo': 0,
            'avg_files_changed_per_pr': 0,
            'avg_lines_changed_per_pr': 0,
            'repos_with_swebench': sum(1 for x in all_info if x.get('swebench')),
            'repos_with_patch': sum(1 for x in all_info if x.get('patch')),
            'repos_with_issue_comments': sum(1 for x in all_info if x.get('issue_comments_count', 0) > 0),
            'total_issue_comments': sum(x.get('issue_comments_count', 0) for x in all_info),
            'avg_issue_comments_per_repo': 0,
            'repos_with_problem_statement': sum(1 for x in all_info 
                if x.get('swebench', {}).get('problem_statement') 
                and x.get('swebench', {}).get('problem_statement') != {})
        }
    }
    
    # Calculate averages
    successful_repos = [x for x in all_info if x.get('success')]
    if successful_repos:
        code_stats_list = [x.get('code_statistics', {}) for x in successful_repos]
        if code_stats_list:
            summary['statistics']['avg_files_per_repo'] = sum(s.get('total_files', 0) for s in code_stats_list) / len(code_stats_list)
            summary['statistics']['avg_lines_per_repo'] = sum(s.get('total_lines', 0) for s in code_stats_list) / len(code_stats_list)
        
        patch_stats_list = [x.get('patch', {}) for x in successful_repos if x.get('patch')]
        if patch_stats_list:
            summary['statistics']['avg_files_changed_per_pr'] = sum(p.get('files_changed', 0) for p in patch_stats_list) / len(patch_stats_list)
            summary['statistics']['avg_lines_changed_per_pr'] = sum(
                (p.get('lines_added', 0) + p.get('lines_deleted', 0)) for p in patch_stats_list
            ) / len(patch_stats_list)
        
        # Calculate average issue comments
        if successful_repos:
            repos_with_comments = [x for x in successful_repos if x.get('issue_comments_count', 0) > 0]
            if repos_with_comments:
                summary['statistics']['avg_issue_comments_per_repo'] = sum(x.get('issue_comments_count', 0) for x in successful_repos) / len(successful_repos)
            else:
                summary['statistics']['avg_issue_comments_per_repo'] = 0
    
    # Create final output
    output_data = {
        'summary': summary,
        'repositories': all_info
    }
    
    # Save to file
    print(f"\nSaving information to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories processed: {summary['total_repositories']}")
    print(f"  ✓ Successful: {summary['successful']}")
    print(f"  ✗ Failed: {summary['failed']}")
    print(f"  ⊘ Skipped (not downloaded): {summary['total_skipped']}")
    if skipped_info:
        print(f"\nSkipped repositories (first 5):")
        for skip in skipped_info[:5]:
            print(f"  - {skip['repo_name']}: {skip['reason']}")
        if len(skipped_info) > 5:
            print(f"  ... and {len(skipped_info) - 5} more")
    print(f"\nStatistics:")
    print(f"  Average files per repo: {summary['statistics']['avg_files_per_repo']:.1f}")
    print(f"  Average lines per repo: {summary['statistics']['avg_lines_per_repo']:.1f}")
    print(f"  Average files changed per PR: {summary['statistics']['avg_files_changed_per_pr']:.1f}")
    print(f"  Average lines changed per PR: {summary['statistics']['avg_lines_changed_per_pr']:.1f}")
    print(f"  Repositories with SWE-Bench data: {summary['statistics']['repos_with_swebench']}")
    print(f"  Repositories with patch data: {summary['statistics']['repos_with_patch']}")
    print(f"  Repositories with issue comments: {summary['statistics']['repos_with_issue_comments']}")
    print(f"  Total issue comments: {summary['statistics']['total_issue_comments']}")
    print(f"  Average issue comments per repo: {summary['statistics']['avg_issue_comments_per_repo']:.2f}")
    print(f"  Repositories with problem_statement: {summary['statistics']['repos_with_problem_statement']}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()

