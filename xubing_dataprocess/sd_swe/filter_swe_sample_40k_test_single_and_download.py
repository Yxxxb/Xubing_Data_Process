from datakit.utils.files import find_all_files
from datakit.utils.tar import read_all_files_in_tar, tar_read_json_file
import argparse
import os
import json
import subprocess
import tempfile
import shutil
import base64
import time
from typing import List, Dict, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed, GitHub API features will be disabled")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


tar_path = "/data/xubing/oss_swe/swebench_sample_40k"


def get_pr_language(tar_file_path: str) -> Optional[str]:
    """Extract the repository language from PR JSON file in tar.
    
    Args:
        tar_file_path (str): Path to the tar file.
        
    Returns:
        Optional[str]: Repository language (e.g., 'Python', 'Java'), or None if not found/error.
    """
    try:
        tar, files_dict = read_all_files_in_tar(tar_file_path, extension='any')
        
        # Find PR JSON file (format: repo#owner#pull#number.json)
        pr_json_file = None
        for filename in files_dict.keys():
            if filename.endswith('.json') and '#pull#' in filename and not filename.endswith('_swebench.json') and '#issuecomment-' not in filename:
                pr_json_file = filename
                break
        
        if pr_json_file is None:
            tar.close()
            return None
        
        # Read PR JSON data
        json_data = tar.extractfile(files_dict[pr_json_file]).read()
        pr_data = json.loads(json_data)
        
        # Get language from repo info (check both head and base repos)
        repo_info = pr_data.get('head', {}).get('repo') or pr_data.get('base', {}).get('repo') or {}
        language = repo_info.get('language')
        
        tar.close()
        return language
        
    except Exception as e:
        print(f"Error extracting language from {os.path.basename(tar_file_path)}: {str(e)}")
        return None


def has_issue_comments(tar_file_path: str) -> bool:
    """Check if a tar file has issue comment files.
    
    Args:
        tar_file_path (str): Path to the tar file.
        
    Returns:
        bool: True if tar file contains issue comment files, False otherwise.
    """
    try:
        tar, files_dict = read_all_files_in_tar(tar_file_path, extension='any')
        
        # Check for issue comment files
        for filename in files_dict.keys():
            if '#issuecomment-' in filename and filename.endswith('.json'):
                tar.close()
                return True
        
        tar.close()
        return False
        
    except Exception:
        return False


def filter_tar_files_by_language(tar_path: str, target_language: str = 'Python', max_files: Optional[int] = None,
                                  require_issue_comments: bool = False) -> List[str]:
    """Filter tar files by repository language.
    
    Args:
        tar_path (str): Directory containing tar files.
        target_language (str): Target language to filter (default: 'Python').
        max_files (int, optional): Maximum number of tar files to process for filtering.
                                   If None, process all files.
        require_issue_comments (bool): If True, only include tar files with issue comments.
    
    Returns:
        List[str]: List of filtered tar file paths.
    """
    print(f"Finding all tar files in {tar_path}...")
    tar_files = find_all_files(tar_path, extension='.tar')
    print(f"Found {len(tar_files)} tar files")
    
    if max_files is not None:
        tar_files = tar_files[:max_files]
        print(f"Limiting filtering to first {max_files} files")
    
    filtered_files = []
    print(f"\nFiltering tar files for language: {target_language}")
    if require_issue_comments:
        print("  Also filtering for tar files with issue comments...")
    
    for tar_file in tqdm(tar_files, desc="Filtering"):
        language = get_pr_language(tar_file)
        if language == target_language:
            # Check issue comments requirement
            if require_issue_comments:
                if has_issue_comments(tar_file):
                    filtered_files.append(tar_file)
            else:
                filtered_files.append(tar_file)
    
    filter_msg = f"with language '{target_language}'"
    if require_issue_comments:
        filter_msg += " and with issue comments"
    print(f"\nFiltered {len(filtered_files)}/{len(tar_files)} tar files {filter_msg}")
    
    return filtered_files


def get_tar_info(tar_file_path: str) -> Dict:
    """Get detailed information about a tar file.
    
    Args:
        tar_file_path (str): Path to the tar file.
        
    Returns:
        dict: Dictionary containing tar file information including:
            - tar_file: Path to tar file
            - pr_json: PR metadata
            - swebench_json: SWE-Bench data
            - patch: Patch content
            - issuecomments: List of issue comment files
            - error: Error message if any
    """
    info = {
        'tar_file': tar_file_path,
        'pr_json': None,
        'swebench_json': None,
        'patch': None,
        'issuecomments': [],
        'error': None
    }
    
    try:
        tar, files_dict = read_all_files_in_tar(tar_file_path, extension='any')
        
        # Find PR JSON file
        for filename in files_dict.keys():
            if filename.endswith('.json') and '#pull#' in filename:
                if filename.endswith('_swebench.json'):
                    # Read swebench JSON
                    json_data = tar.extractfile(files_dict[filename]).read()
                    info['swebench_json'] = json.loads(json_data)
                elif '#issuecomment-' not in filename:
                    # Read PR JSON
                    json_data = tar.extractfile(files_dict[filename]).read()
                    info['pr_json'] = json.loads(json_data)
        
        # Find patch file
        for filename in files_dict.keys():
            if filename.endswith('.patch'):
                patch_data = tar.extractfile(files_dict[filename]).read()
                info['patch'] = patch_data.decode('utf-8', errors='ignore')
                break
        
        # Find issue comment files
        for filename in files_dict.keys():
            if '#issuecomment-' in filename and filename.endswith('.json'):
                json_data = tar.extractfile(files_dict[filename]).read()
                info['issuecomments'].append({
                    'filename': filename,
                    'data': json.loads(json_data)
                })
        
        tar.close()
        
    except Exception as e:
        info['error'] = str(e)
    
    return info


def print_tar_info(info: Dict) -> None:
    """Print detailed information about a tar file.
    
    Args:
        info (dict): Information dictionary from get_tar_info.
    """
    print("\n" + "="*80)
    print(f"TAR FILE INFORMATION: {os.path.basename(info['tar_file'])}")
    print("="*80)
    
    if info['error']:
        print(f"\nError: {info['error']}")
        return
    
    # PR JSON information
    if info['pr_json']:
        pr = info['pr_json']
        print(f"\nPR Information:")
        print(f"  Title: {pr.get('title', 'N/A')}")
        print(f"  Number: {pr.get('number', 'N/A')}")
        print(f"  State: {pr.get('state', 'N/A')}")
        print(f"  Created: {pr.get('created_at', 'N/A')}")
        print(f"  Repository: {pr.get('head', {}).get('repo', {}).get('full_name', 'N/A')}")
        print(f"  Language: {pr.get('head', {}).get('repo', {}).get('language', 'N/A')}")
        print(f"  Additions: {pr.get('additions', 'N/A')}")
        print(f"  Deletions: {pr.get('deletions', 'N/A')}")
        print(f"  Changed Files: {pr.get('changed_files', 'N/A')}")
    else:
        print("\nPR JSON: Not found")
    
    # SWE-Bench JSON information
    if info['swebench_json']:
        swe = info['swebench_json']
        print(f"\nSWE-Bench Information:")
        print(f"  Instance ID: {swe.get('instance_id', 'N/A')}")
        print(f"  Repository: {swe.get('repo', 'N/A')}")
        print(f"  Base Commit: {swe.get('base_commit', 'N/A')}")
        print(f"  Edit Files: {swe.get('edit_files', [])}")
        print(f"  Oracle Files: {swe.get('oracle_files', [])}")
        if swe.get('patch'):
            patch_preview = swe['patch'][:200] + '...' if len(swe.get('patch', '')) > 200 else swe.get('patch', '')
            print(f"  Patch Preview: {patch_preview}")
    else:
        print("\nSWE-Bench JSON: Not found")
    
    # Patch file
    if info['patch']:
        patch_preview = info['patch'][:300] + '...' if len(info['patch']) > 300 else info['patch']
        print(f"\nPatch File Preview:\n{patch_preview}")
    else:
        print("\nPatch File: Not found")
    
    # Issue comments
    if info['issuecomments']:
        print(f"\nIssue Comments: {len(info['issuecomments'])} found")
        for idx, comment in enumerate(info['issuecomments'][:5], 1):  # Show first 5
            comment_data = comment['data']
            print(f"  {idx}. {comment['filename']}")
            print(f"     Author: {comment_data.get('user', {}).get('login', 'N/A')}")
            print(f"     Created: {comment_data.get('created_at', 'N/A')}")
            body_preview = comment_data.get('body', '')[:100] + '...' if len(comment_data.get('body', '')) > 100 else comment_data.get('body', '')
            print(f"     Body: {body_preview}")
        if len(info['issuecomments']) > 5:
            print(f"  ... and {len(info['issuecomments']) - 5} more")
    else:
        print("\nIssue Comments: None found")
    
    print("="*80)


def get_github_commit_info_via_api(repo: str, commit_sha: str, github_token: Optional[str] = None) -> Optional[Dict]:
    """Get commit information from GitHub API.
    
    Args:
        repo (str): Repository name (format: 'owner/repo' or '/owner/repo').
        commit_sha (str): Commit SHA.
        github_token (str, optional): GitHub personal access token for higher rate limits.
        
    Returns:
        Optional[Dict]: Commit information dictionary, or None if error.
    """
    if not HAS_REQUESTS:
        print("Error: requests library not installed")
        return None
    
    # Clean repo name (remove leading / if present)
    repo = repo.lstrip('/')
    
    url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'SWE-Bench-Filter-Tool'
    }
    
    if github_token and github_token.strip() and github_token.upper() not in ['YOUR_TOKEN', 'TOKEN', 'GITHUB_TOKEN']:
        if github_token.startswith('ghp_') or github_token.startswith('github_pat_'):
            headers['Authorization'] = f'Bearer {github_token}'
        else:
            headers['Authorization'] = f'token {github_token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Check for rate limit
        if response.status_code == 403:
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
            rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'N/A')
            if rate_limit_remaining == '0' or (isinstance(rate_limit_remaining, str) and rate_limit_remaining.isdigit() and int(rate_limit_remaining) == 0):
                error_msg = response.json().get('message', 'Rate limit exceeded')
                print(f"⚠ Rate limit exceeded! Remaining: {rate_limit_remaining}, Reset at: {rate_limit_reset}")
                print(f"  Consider using --github_token to increase rate limit to 5000/hour")
                return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching commit info from GitHub API: {str(e)}")
        return None


def get_github_file_tree(repo: str, tree_sha: str, github_token: Optional[str] = None, recursive: bool = True) -> Optional[Dict]:
    """Get file tree from GitHub API.
    
    Args:
        repo (str): Repository name (format: 'owner/repo' or '/owner/repo').
        tree_sha (str): Tree SHA.
        github_token (str, optional): GitHub personal access token.
        recursive (bool): If True, get recursive tree.
        
    Returns:
        Optional[Dict]: Tree information dictionary, or None if error.
    """
    if not HAS_REQUESTS:
        return None
    
    repo = repo.lstrip('/')
    url = f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}"
    
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'SWE-Bench-Filter-Tool'
    }
    
    if github_token and github_token.strip() and github_token.upper() not in ['YOUR_TOKEN', 'TOKEN', 'GITHUB_TOKEN']:
        if github_token.startswith('ghp_') or github_token.startswith('github_pat_'):
            headers['Authorization'] = f'Bearer {github_token}'
        else:
            headers['Authorization'] = f'token {github_token}'
    
    params = {'recursive': '1'} if recursive else {}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Check for rate limit
        if response.status_code == 403:
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
            if rate_limit_remaining == '0' or (isinstance(rate_limit_remaining, str) and rate_limit_remaining.isdigit() and int(rate_limit_remaining) == 0):
                print(f"⚠ Rate limit exceeded when fetching file tree. Consider using --github_token")
                return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file tree: {str(e)}")
        return None


def get_github_file_blob(repo: str, blob_sha: str, github_token: Optional[str] = None) -> Optional[bytes]:
    """Get file content (blob) from GitHub API.
    
    Args:
        repo (str): Repository name (format: 'owner/repo' or '/owner/repo').
        blob_sha (str): Blob SHA.
        github_token (str, optional): GitHub personal access token.
        
    Returns:
        Optional[bytes]: File content as bytes, or None if error.
    """
    if not HAS_REQUESTS:
        return None
    
    repo = repo.lstrip('/')
    url = f"https://api.github.com/repos/{repo}/git/blobs/{blob_sha}"
    
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'SWE-Bench-Filter-Tool'
    }
    
    if github_token and github_token.strip() and github_token.upper() not in ['YOUR_TOKEN', 'TOKEN', 'GITHUB_TOKEN']:
        if github_token.startswith('ghp_') or github_token.startswith('github_pat_'):
            headers['Authorization'] = f'Bearer {github_token}'
        else:
            headers['Authorization'] = f'token {github_token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Check for rate limit
        if response.status_code == 403:
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'N/A')
            if rate_limit_remaining == '0' or (isinstance(rate_limit_remaining, str) and rate_limit_remaining.isdigit() and int(rate_limit_remaining) == 0):
                print(f"⚠ Rate limit exceeded when fetching blob. Consider using --github_token")
                return None
        
        response.raise_for_status()
        blob_data = response.json()
        
        # Decode base64 content
        content = blob_data.get('content', '')
        encoding = blob_data.get('encoding', 'base64')
        
        if encoding == 'base64':
            # GitHub API returns base64 encoded content with newlines
            content = content.replace('\n', '')
            return base64.b64decode(content)
        else:
            # Some files might be returned as-is
            return content.encode('utf-8') if isinstance(content, str) else content
            
    except Exception as e:
        print(f"Error fetching blob {blob_sha}: {str(e)}")
        return None


def download_commit_code_via_api(repo: str, commit_sha: str, output_dir: str, 
                                  github_token: Optional[str] = None,
                                  max_files: Optional[int] = None,
                                  max_time_minutes: Optional[float] = None) -> Dict:
    """Download all source code and information for a specific commit via GitHub API.
    
    Args:
        repo (str): Repository name (format: 'owner/repo' or '/owner/repo').
        commit_sha (str): Commit SHA.
        output_dir (str): Directory to save files.
        github_token (str, optional): GitHub personal access token.
        
    Returns:
        Dict: Result dictionary with stats and error information.
    """
    result = {
        'success': False,
        'files_downloaded': 0,
        'total_files': 0,
        'error': None,
        'skipped': False,
        'output_dir': output_dir
    }
    
    if not HAS_REQUESTS:
        result['error'] = "requests library not installed"
        return result
    
    repo_clean = repo.lstrip('/')
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Get commit information
        print(f"Fetching commit information for {commit_sha}...")
        commit_info = get_github_commit_info_via_api(repo, commit_sha, github_token)
        if not commit_info:
            result['error'] = "Failed to fetch commit information"
            return result
        
        # Save commit info
        commit_info_file = os.path.join(output_dir, 'commit_info.json')
        with open(commit_info_file, 'w', encoding='utf-8') as f:
            json.dump(commit_info, f, indent=2, ensure_ascii=False)
        
        # 2. Get file tree
        tree_sha = commit_info['commit']['tree']['sha']
        print(f"Fetching file tree (tree SHA: {tree_sha})...")
        tree_info = get_github_file_tree(repo, tree_sha, github_token, recursive=True)
        if not tree_info:
            result['error'] = "Failed to fetch file tree"
            return result
        
        # Save tree info
        tree_info_file = os.path.join(output_dir, 'tree_info.json')
        with open(tree_info_file, 'w', encoding='utf-8') as f:
            json.dump(tree_info, f, indent=2, ensure_ascii=False)
        
        # 3. Filter files (only files, not directories, and exclude .git files)
        files = [item for item in tree_info.get('tree', []) 
                if item['type'] == 'blob' and not item['path'].startswith('.git/')]
        
        result['total_files'] = len(files)
        print(f"Found {len(files)} files to download")
        
        # Check if file count exceeds threshold (before downloading)
        if max_files is not None and len(files) > max_files:
            result['error'] = f"File count ({len(files)}) exceeds threshold ({max_files})"
            result['skipped'] = True
            print(f"⚠ Skipping: File count ({len(files)}) exceeds threshold ({max_files})")
            return result
        
        # Estimate API calls needed
        # Each commit: 1 commit API call + 1 tree API call + N blob API calls
        estimated_calls = 2 + len(files)
        print(f"Estimated API calls for this commit: ~{estimated_calls}")
        
        if not github_token:
            print(f"⚠ Warning: Without token, rate limit is 60 calls/hour")
            print(f"  This commit alone requires ~{estimated_calls} calls")
            print(f"  If rate limited, download will fail. Consider using --github_token")
        
        # 4. Download each file
        source_code_dir = os.path.join(output_dir, 'source_code')
        os.makedirs(source_code_dir, exist_ok=True)
        
        failed_files = []
        
        for file_item in tqdm(files, desc="Downloading files"):
            file_path = file_item['path']
            blob_sha = file_item['sha']
            
            # Skip very large files (GitHub API has limits)
            size = file_item.get('size', 0)
            if size > 100 * 1024 * 1024:  # Skip files larger than 100MB
                print(f"\nSkipping large file: {file_path} ({size / 1024 / 1024:.1f}MB)")
                failed_files.append({'path': file_path, 'reason': 'file too large'})
                continue
            
            try:
                # Get file content
                content = get_github_file_blob(repo, blob_sha, github_token)
                if content is None:
                    failed_files.append({'path': file_path, 'reason': 'failed to fetch blob'})
                    continue
                
                # Create directory structure
                full_file_path = os.path.join(source_code_dir, file_path)
                file_dir = os.path.dirname(full_file_path)
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                
                # Save file
                # Try to decode as text, fallback to binary
                try:
                    text_content = content.decode('utf-8')
                    with open(full_file_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                except UnicodeDecodeError:
                    # Binary file, save as-is
                    with open(full_file_path, 'wb') as f:
                        f.write(content)
                
                result['files_downloaded'] += 1
                
            except Exception as e:
                failed_files.append({'path': file_path, 'reason': str(e)})
        
        # Save failed files info
        if failed_files:
            failed_file = os.path.join(output_dir, 'failed_files.json')
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_files, f, indent=2, ensure_ascii=False)
            print(f"\nWarning: {len(failed_files)} files failed to download")
        
        # Save summary
        summary = {
            'repo': repo,
            'commit_sha': commit_sha,
            'total_files': result['total_files'],
            'files_downloaded': result['files_downloaded'],
            'failed_files_count': len(failed_files),
            'download_time': None  # Could add timestamp
        }
        
        summary_file = os.path.join(output_dir, 'download_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        result['success'] = True
        print(f"\nDownload completed: {result['files_downloaded']}/{result['total_files']} files")
        print(f"Files saved to: {output_dir}")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def get_github_source_via_git_clone(repo: str, commit_sha: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Clone repository and checkout specific commit using git.
    
    Args:
        repo (str): Repository name (format: 'owner/repo' or '/owner/repo').
        commit_sha (str): Commit SHA to checkout.
        output_dir (str, optional): Directory to clone to. If None, uses temporary directory.
        
    Returns:
        Optional[str]: Path to cloned repository directory, or None if error.
    """
    # Clean repo name
    repo = repo.lstrip('/')
    repo_url = f"https://github.com/{repo}.git"
    
    # Create output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='github_clone_')
        output_dir = temp_dir
    
    repo_dir = os.path.join(output_dir, repo.replace('/', '_'))
    
    try:
        # Clone repository (shallow clone for faster operation)
        print(f"Cloning repository {repo} to {repo_dir}...")
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, repo_dir], 
                      check=True, capture_output=True, text=True)
        
        # Fetch the specific commit (since we did shallow clone)
        subprocess.run(['git', 'fetch', 'origin', commit_sha], 
                      cwd=repo_dir, check=True, capture_output=True, text=True)
        
        # Checkout the specific commit
        print(f"Checking out commit {commit_sha}...")
        subprocess.run(['git', 'checkout', commit_sha], 
                      cwd=repo_dir, check=True, capture_output=True, text=True)
        
        print(f"Successfully cloned and checked out commit {commit_sha}")
        print(f"Repository located at: {repo_dir}")
        
        return repo_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {str(e)}")
        if os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
        return None


def download_commit_code_for_tar(tar_file_path: str, output_base_dir: str, 
                                   github_token: Optional[str] = None,
                                   max_files: Optional[int] = None) -> Dict:
    """Download commit code for a tar file.
    
    Args:
        tar_file_path (str): Path to tar file.
        output_base_dir (str): Base directory to save files. Will create subdirectory with tar name.
        github_token (str, optional): GitHub token.
        
    Returns:
        Dict: Result dictionary.
    """
    result = {
        'tar_file': tar_file_path,
        'success': False,
        'error': None
    }
    
    try:
        # Get tar info to extract repo and base_commit
        info = get_tar_info(tar_file_path)
        
        if info['error']:
            result['error'] = f"Failed to read tar file: {info['error']}"
            return result
        
        if not info['swebench_json']:
            result['error'] = "SWE-Bench JSON not found in tar file"
            return result
        
        swebench_data = info['swebench_json']
        repo = swebench_data.get('repo', '')
        base_commit = swebench_data.get('base_commit', '')
        
        if not repo or not base_commit:
            result['error'] = f"Missing repo or base_commit. repo={repo}, base_commit={base_commit}"
            return result
        
        # Create output directory named after tar file (without .tar extension)
        tar_basename = os.path.basename(tar_file_path)
        tar_name_without_ext = os.path.splitext(tar_basename)[0]
        output_dir = os.path.join(output_base_dir, tar_name_without_ext)
        
        # Download commit code
        download_result = download_commit_code_via_api(repo, base_commit, output_dir, 
                                                       github_token, max_files=max_files)
        
        result['success'] = download_result['success']
        result['error'] = download_result.get('error')
        result['files_downloaded'] = download_result.get('files_downloaded', 0)
        result['total_files'] = download_result.get('total_files', 0)
        result['skipped'] = download_result.get('skipped', False)
        result['output_dir'] = output_dir
        result['repo'] = repo
        result['base_commit'] = base_commit
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def get_commit_source_code(tar_file_path: str, method: str = 'git', output_dir: Optional[str] = None, 
                           github_token: Optional[str] = None) -> Optional[Dict]:
    """Get source code for base_commit from a tar file.
    
    Args:
        tar_file_path (str): Path to tar file.
        method (str): Method to use ('git' or 'api'). Default: 'git'.
        output_dir (str, optional): Directory for git clone (only for 'git' method).
        github_token (str, optional): GitHub token for API (only for 'api' method).
        
    Returns:
        Optional[Dict]: Dictionary containing:
            - repo: Repository name
            - base_commit: Commit SHA
            - commit_info: Commit information (for API method)
            - repo_path: Path to cloned repository (for git method)
            - error: Error message if any
    """
    result = {
        'repo': None,
        'base_commit': None,
        'commit_info': None,
        'repo_path': None,
        'error': None
    }
    
    try:
        # Get swebench JSON to extract repo and base_commit
        info = get_tar_info(tar_file_path)
        
        if info['error']:
            result['error'] = f"Failed to read tar file: {info['error']}"
            return result
        
        if not info['swebench_json']:
            result['error'] = "SWE-Bench JSON not found in tar file"
            return result
        
        swebench_data = info['swebench_json']
        repo = swebench_data.get('repo', '')
        base_commit = swebench_data.get('base_commit', '')
        
        if not repo or not base_commit:
            result['error'] = f"Missing repo or base_commit. repo={repo}, base_commit={base_commit}"
            return result
        
        result['repo'] = repo
        result['base_commit'] = base_commit
        
        if method == 'api':
            # Use GitHub API
            commit_info = get_github_commit_info_via_api(repo, base_commit, github_token)
            result['commit_info'] = commit_info
            if commit_info is None:
                result['error'] = "Failed to fetch commit info from GitHub API"
        
        elif method == 'git':
            # Use git clone
            repo_path = get_github_source_via_git_clone(repo, base_commit, output_dir)
            result['repo_path'] = repo_path
            if repo_path is None:
                result['error'] = "Failed to clone repository"
        
        else:
            result['error'] = f"Unknown method: {method}. Use 'git' or 'api'"
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    parser = argparse.ArgumentParser(description='Filter and inspect SWE-Bench tar files by language')
    parser.add_argument('--tar_path', type=str, default=tar_path,
                        help='Path to directory containing tar files')
    parser.add_argument('--language', type=str, default='Python',
                        help='Language to filter (default: Python)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of tar files to process for filtering')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file to save filtered file paths')
    parser.add_argument('--inspect_first', action='store_true',
                        help='Inspect the first filtered tar file')
    parser.add_argument('--get_source', action='store_true',
                        help='Get source code for base_commit from first filtered tar file')
    parser.add_argument('--source_method', type=str, default='git', choices=['git', 'api'],
                        help='Method to get source code: git (clone) or api (GitHub API)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for git clone (only for git method)')
    parser.add_argument('--github_token', type=str, default=None,
                        help='GitHub personal access token (for API method). Can also set via GITHUB_TOKEN env var. Get token from: https://github.com/settings/tokens')
    parser.add_argument('--download_commits', action='store_true',
                        help='Download commit code for all filtered tar files')
    parser.add_argument('--commit_output_base', type=str, 
                        default='/data/xubing/oss_swe/swebench_sample_40k_commit_raw_code',
                        help='Base directory to save downloaded commit code (default: /data/xubing/oss_swe/swebench_sample_40k_commit_raw_code)')
    parser.add_argument('--download_max', type=int, default=None,
                        help='Maximum number of tar files to download commit code for (for testing)')
    parser.add_argument('--max_files_per_commit', type=int, default=500,
                        help='Maximum number of files per commit to download. Commits exceeding this will be skipped (default: 500)')
    parser.add_argument('--require_issue_comments', action='store_true',
                        help='Only download tar files that have issue comments')
    
    args = parser.parse_args()
    
    # Get GitHub token from environment variable if not provided via command line
    if args.github_token is None:
        args.github_token = os.environ.get('GITHUB_TOKEN')
    
    # Filter tar files by language (and optionally by issue comments)
    filtered_files = filter_tar_files_by_language(
        args.tar_path, 
        target_language=args.language,
        max_files=args.max_files,
        require_issue_comments=args.require_issue_comments
    )
    
    if not filtered_files:
        print("\nNo files found matching the criteria.")
        return
    
    # Save filtered file paths
    if args.output:
        print(f"\nSaving filtered file paths to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(filtered_files, f, indent=2)
        print(f"Saved {len(filtered_files)} file paths")
    
    # Inspect first tar file if requested
    if args.inspect_first:
        print(f"\nInspecting first filtered tar file: {os.path.basename(filtered_files[0])}")
        info = get_tar_info(filtered_files[0])
        print_tar_info(info)
    
    # Get source code for first tar file if requested
    if args.get_source:
        print(f"\nGetting source code for first filtered tar file: {os.path.basename(filtered_files[0])}")
        source_result = get_commit_source_code(
            filtered_files[0],
            method=args.source_method,
            output_dir=args.output_dir,
            github_token=args.github_token
        )
        
        if source_result['error']:
            print(f"\nError: {source_result['error']}")
        else:
            print(f"\nSuccess!")
            print(f"  Repository: {source_result['repo']}")
            print(f"  Base Commit: {source_result['base_commit']}")
            if args.source_method == 'api' and source_result['commit_info']:
                commit_info = source_result['commit_info']
                print(f"  Commit Author: {commit_info.get('commit', {}).get('author', {}).get('name', 'N/A')}")
                print(f"  Commit Message: {commit_info.get('commit', {}).get('message', 'N/A')[:100]}...")
                print(f"  Commit URL: {commit_info.get('html_url', 'N/A')}")
            elif args.source_method == 'git' and source_result['repo_path']:
                print(f"  Repository cloned to: {source_result['repo_path']}")
    
    # Download commit code for all filtered files if requested
    if args.download_commits:
        print(f"\n{'='*80}")
        print("DOWNLOADING COMMIT CODE")
        print(f"{'='*80}")
        print(f"Output base directory: {args.commit_output_base}")
        print(f"Total tar files to process: {len(filtered_files)}")
        
        # Check if GitHub token is provided
        if args.github_token:
            print(f"✓ Using GitHub token (higher rate limit: 5000 requests/hour)")
        else:
            print(f"⚠ No GitHub token provided - using unauthenticated API")
            print(f"  Rate limit: 60 requests/hour per IP address")
            print(f"  Recommendation: Use --github_token for faster downloads")
            print(f"  Get token from: https://github.com/settings/tokens")
        
        # Create base directory
        os.makedirs(args.commit_output_base, exist_ok=True)
        
        # Limit number of files if specified
        files_to_download = filtered_files
        if args.download_max is not None:
            files_to_download = filtered_files[:args.download_max]
            print(f"Limiting to first {args.download_max} files")
        
        # Download statistics
        download_stats = {
            'total': len(files_to_download),
            'success': 0,
            'failed': 0,
            'skipped_already_exists': 0,  # Already downloaded
            'skipped_file_count': 0,  # Skipped due to file count threshold
            'skipped_files': [],  # List of skipped files with reasons
            'errors': []
        }
        
        print(f"\nFile count threshold: {args.max_files_per_commit} files per commit")
        print(f"Starting download for {len(files_to_download)} tar files...")
        
        skipped_files_list = []  # Track skipped files
        
        for idx, tar_file in enumerate(tqdm(files_to_download, desc="Downloading commits"), 1):
            tar_basename = os.path.basename(tar_file)
            print(f"\n[{idx}/{len(files_to_download)}] Processing: {tar_basename}")
            
            # Check if already downloaded
            tar_name_without_ext = os.path.splitext(tar_basename)[0]
            output_dir = os.path.join(args.commit_output_base, tar_name_without_ext)
            summary_file = os.path.join(output_dir, 'download_summary.json')
            
            if os.path.exists(summary_file):
                print(f"  Already downloaded, skipping...")
                download_stats['skipped_already_exists'] += 1
                continue
            
            # Download commit code
            result = download_commit_code_for_tar(
                tar_file, 
                args.commit_output_base,
                github_token=args.github_token,
                max_files=args.max_files_per_commit
            )
            
            if result.get('skipped', False):
                download_stats['skipped_file_count'] += 1
                skipped_info = {
                    'tar_file': tar_basename,
                    'repo': result.get('repo', 'N/A'),
                    'base_commit': result.get('base_commit', 'N/A'),
                    'file_count': result.get('total_files', 0),
                    'reason': result.get('error', 'File count exceeds threshold')
                }
                skipped_files_list.append(skipped_info)
                download_stats['skipped_files'].append(skipped_info)
                print(f"  ⊘ Skipped: {result.get('error', 'File count exceeds threshold')}")
            elif result['success']:
                download_stats['success'] += 1
                print(f"  ✓ Success: {result.get('files_downloaded', 0)}/{result.get('total_files', 0)} files")
            else:
                download_stats['failed'] += 1
                error_msg = result.get('error', 'Unknown error')
                download_stats['errors'].append({
                    'tar_file': tar_basename,
                    'error': error_msg
                })
                print(f"  ✗ Failed: {error_msg}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        print(f"Total processed: {download_stats['total']}")
        print(f"  ✓ Successful: {download_stats['success']}")
        print(f"  ✗ Failed: {download_stats['failed']}")
        print(f"  ⊘ Skipped (already exists): {download_stats['skipped_already_exists']}")
        print(f"  ⊘ Skipped (file count > {args.max_files_per_commit}): {download_stats['skipped_file_count']}")
        
        if download_stats['skipped_files']:
            print(f"\nSkipped files (file count > {args.max_files_per_commit}): {len(download_stats['skipped_files'])}")
            for skip_info in download_stats['skipped_files'][:5]:  # Show first 5
                print(f"  - {skip_info['tar_file']}: {skip_info['file_count']} files")
            if len(download_stats['skipped_files']) > 5:
                print(f"  ... and {len(download_stats['skipped_files']) - 5} more")
        
        if download_stats['errors']:
            print(f"\nErrors ({len(download_stats['errors'])}):")
            for error_info in download_stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error_info['tar_file']}: {error_info['error']}")
            if len(download_stats['errors']) > 10:
                print(f"  ... and {len(download_stats['errors']) - 10} more errors")
        
        # Save download statistics
        stats_file = os.path.join(args.commit_output_base, 'download_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(download_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {stats_file}")
        
        # Save skipped files separately
        skipped_file = os.path.join(args.commit_output_base, 'skipped_files.json')
        with open(skipped_file, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': args.max_files_per_commit,
                'total_skipped': len(skipped_files_list),
                'skipped_files': skipped_files_list
            }, f, indent=2, ensure_ascii=False)
        if skipped_files_list:
            print(f"Skipped files list saved to: {skipped_file}")


if __name__ == '__main__':
    main()
