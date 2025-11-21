from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file 
from datakit.utils.files import mem_efficient_read_jsonl_file, read_parquet_file, find_all_files_multi_folders
from datakit.utils.distributed import dist_split_files, get_distributed_env, barrier_all_processes
from datakit.utils.mp import multi_process_with_append
from datakit.utils.tar import read_all_files_in_tar, tar_read_jsonl_file, tar_read_json_file
import argparse
import os
import json
from collections import defaultdict, Counter
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        return iterable

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import gridspec
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualization will be disabled")


tar_path = "/data/xubing/oss_swe/swebench_sample_40k"


def inspect_tar_file(tar_file_path: str, collect_no_ext_files: bool = False) -> dict:
    """Inspect a single tar file and return statistics.
    
    Args:
        tar_file_path (str): Path to the tar file.
        collect_no_ext_files (bool): If True, collect filenames without extensions.
        
    Returns:
        dict: Statistics about the tar file content.
    """
    stats = {
        'tar_file': tar_file_path,
        'total_files': 0,
        'file_types': {},
        'has_pull_json': False,
        'has_patch': False,
        'has_swebench_json': False,
        'has_issuecomment': False,
        'no_ext_files': [] if collect_no_ext_files else None,
        'error': None
    }
    file_types = defaultdict(int)
    
    try:
        tar, files_dict = read_all_files_in_tar(tar_file_path, extension='any')
        stats['total_files'] = len(files_dict)
        
        # Analyze file types and patterns
        for filename in files_dict.keys():
            # Count by extension
            ext = os.path.splitext(filename)[1]
            file_types[ext] += 1
            
            # Collect files without extensions
            if collect_no_ext_files and not ext:
                stats['no_ext_files'].append(filename)
            
            # Check for specific file patterns
            if filename.endswith('.json') and '#pull#' in filename:
                if filename.endswith('_swebench.json'):
                    stats['has_swebench_json'] = True
                else:
                    stats['has_pull_json'] = True
            
            if filename.endswith('.patch'):
                stats['has_patch'] = True
            
            if '#issuecomment-' in filename:
                stats['has_issuecomment'] = True
        
        stats['file_types'] = dict(file_types)
        tar.close()
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def inspect_all_tar_files(tar_path: str, max_files: int = None) -> list:
    """Inspect all tar files in the given path.
    
    Args:
        tar_path (str): Directory containing tar files.
        max_files (int, optional): Maximum number of files to inspect. 
                                   If None, inspect all files.
    
    Returns:
        list: List of statistics dictionaries for each tar file.
    """
    print(f"Finding all tar files in {tar_path}...")
    tar_files = find_all_files(tar_path, extension='.tar')
    print(f"Found {len(tar_files)} tar files")
    
    if max_files is not None:
        tar_files = tar_files[:max_files]
        print(f"Limiting to first {max_files} files")
    
    all_stats = []
    for tar_file in tqdm(tar_files, desc="Inspecting tar files"):
        stats = inspect_tar_file(tar_file)
        all_stats.append(stats)
    
    return all_stats


def print_statistics(all_stats: list) -> None:
    """Print summary statistics from all inspected tar files.
    
    Args:
        all_stats (list): List of statistics dictionaries.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_tars = len(all_stats)
    successful = sum(1 for s in all_stats if s['error'] is None)
    failed = total_tars - successful
    
    print(f"\nTotal tar files inspected: {total_tars}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for stats in all_stats:
            if stats['error']:
                print(f"  - {os.path.basename(stats['tar_file'])}: {stats['error']}")
    
    # Aggregate statistics
    total_files_in_tars = sum(s['total_files'] for s in all_stats if s['error'] is None)
    avg_files_per_tar = total_files_in_tars / successful if successful > 0 else 0
    
    print(f"\nFile statistics (from successful inspections):")
    print(f"  - Total files in all tars: {total_files_in_tars}")
    print(f"  - Average files per tar: {avg_files_per_tar:.2f}")
    
    # File type distribution
    file_type_counts = defaultdict(int)
    for stats in all_stats:
        if stats['error'] is None:
            for ext, count in stats['file_types'].items():
                file_type_counts[ext] += count
    
    if file_type_counts:
        print(f"\nFile type distribution (total counts):")
        for ext, count in sorted(file_type_counts.items(), key=lambda x: -x[1]):
            print(f"  - {ext or '(no extension)'}: {count}")
        
        # Show sample files without extensions if any
        no_ext_count = file_type_counts.get('', 0)
        if no_ext_count > 0:
            print(f"\n  Sample files without extension (showing first 10):")
            sample_no_ext = []
            for stats in all_stats:
                if stats.get('no_ext_files'):
                    sample_no_ext.extend(stats['no_ext_files'][:3])  # Get 3 from each tar
                    if len(sample_no_ext) >= 10:
                        break
            for fname in sample_no_ext[:10]:
                print(f"    - {fname}")
    
    # Pattern presence statistics
    has_pull_json = sum(1 for s in all_stats if s.get('has_pull_json', False))
    has_patch = sum(1 for s in all_stats if s.get('has_patch', False))
    has_swebench = sum(1 for s in all_stats if s.get('has_swebench_json', False))
    has_issuecomment = sum(1 for s in all_stats if s.get('has_issuecomment', False))
    
    print(f"\nPattern presence (number of tars containing each pattern):")
    print(f"  - Has pull#*.json: {has_pull_json}/{successful} ({100*has_pull_json/successful:.1f}%)" if successful > 0 else "  - Has pull#*.json: 0")
    print(f"  - Has .patch files: {has_patch}/{successful} ({100*has_patch/successful:.1f}%)" if successful > 0 else "  - Has .patch files: 0")
    print(f"  - Has _swebench.json: {has_swebench}/{successful} ({100*has_swebench/successful:.1f}%)" if successful > 0 else "  - Has _swebench.json: 0")
    print(f"  - Has issuecomment files: {has_issuecomment}/{successful} ({100*has_issuecomment/successful:.1f}%)" if successful > 0 else "  - Has issuecomment files: 0")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Inspect and statistics tar files in swebench sample')
    parser.add_argument('--tar_path', type=str, default=tar_path,
                        help='Path to directory containing tar files')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of tar files to inspect (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file to save statistics (optional)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly sample N files to inspect (optional)')
    parser.add_argument('--show_no_ext', action='store_true',
                        help='Show sample filenames without extensions')
    
    args = parser.parse_args()
    
    # Find tar files
    tar_files = find_all_files(args.tar_path, extension='.tar')
    
    if args.sample is not None and args.sample < len(tar_files):
        import random
        tar_files = random.sample(tar_files, args.sample)
        print(f"Randomly sampled {args.sample} files from {len(find_all_files(args.tar_path, extension='.tar'))} total files")
    
    if args.max_files is not None:
        tar_files = tar_files[:args.max_files]
    
    print(f"Inspecting {len(tar_files)} tar files...")
    
    # Inspect all tar files
    all_stats = []
    for tar_file in tqdm(tar_files, desc="Inspecting"):
        stats = inspect_tar_file(tar_file, collect_no_ext_files=args.show_no_ext)
        all_stats.append(stats)
    
    # Print statistics
    print_statistics(all_stats)
    
    # Save to file if requested
    if args.output:
        print(f"\nSaving statistics to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print("Done!")


def check_oracle_files_empty_ratio(tar_path: str, max_files: int = None) -> None:
    """Check the ratio of swebench.json files with empty oracle_files list.
    
    Args:
        tar_path (str): Directory containing tar files.
        max_files (int, optional): Maximum number of tar files to check.
    """
    print(f"Finding all tar files in {tar_path}...")
    tar_files = find_all_files(tar_path, extension='.tar')
    print(f"Found {len(tar_files)} tar files")
    
    if max_files is not None:
        tar_files = tar_files[:max_files]
        print(f"Limiting to first {max_files} files")
    
    total_swebench_files = 0
    empty_oracle_files = 0
    error_count = 0
    error_details = []
    
    print(f"\nChecking oracle_files in _swebench.json files...")
    
    for tar_file in tqdm(tar_files, desc="Processing"):
        try:
            tar, files_dict = read_all_files_in_tar(tar_file, extension='any')
            
            # Find all _swebench.json files
            for filename, member in files_dict.items():
                if filename.endswith('_swebench.json'):
                    total_swebench_files += 1
                    
                    try:
                        # Read and parse JSON
                        json_data = tar.extractfile(member).read()
                        swebench_data = json.loads(json_data)
                        
                        # Check oracle_files field
                        oracle_files = swebench_data.get('oracle_files', None)
                        if oracle_files is None:
                            print(f"Warning: {filename} in {os.path.basename(tar_file)} has no 'oracle_files' field")
                        elif isinstance(oracle_files, list) and len(oracle_files) == 0:
                            empty_oracle_files += 1
                        elif isinstance(oracle_files, list):
                            # oracle_files is not empty
                            pass
                        else:
                            print(f"Warning: {filename} in {os.path.basename(tar_file)} has oracle_files of type {type(oracle_files)}")
                    except json.JSONDecodeError as e:
                        error_count += 1
                        error_details.append(f"{os.path.basename(tar_file)}/{filename}: JSON decode error - {str(e)}")
                    except Exception as e:
                        error_count += 1
                        error_details.append(f"{os.path.basename(tar_file)}/{filename}: {str(e)}")
            
            tar.close()
        except Exception as e:
            print(f"Error processing {os.path.basename(tar_file)}: {str(e)}")
            error_count += 1
    
    # Print statistics
    print("\n" + "="*80)
    print("ORACLE_FILES STATISTICS")
    print("="*80)
    print(f"\nTotal _swebench.json files found: {total_swebench_files}")
    print(f"Files with empty oracle_files list: {empty_oracle_files}")
    print(f"Files with non-empty oracle_files list: {total_swebench_files - empty_oracle_files}")
    
    if total_swebench_files > 0:
        empty_ratio = empty_oracle_files / total_swebench_files * 100
        print(f"\nEmpty oracle_files ratio: {empty_oracle_files}/{total_swebench_files} = {empty_ratio:.2f}%")
    else:
        print("\nNo _swebench.json files found!")
    
    if error_count > 0:
        print(f"\nErrors encountered: {error_count}")
        if len(error_details) <= 20:
            print("Error details:")
            for detail in error_details:
                print(f"  - {detail}")
        else:
            print(f"Error details (showing first 20 of {len(error_details)}):")
            for detail in error_details[:20]:
                print(f"  - {detail}")
    
    print("="*80)


def collect_pr_statistics(tar_path: str, max_files: int = None) -> dict:
    """Collect PR statistics from tar files.
    
    Args:
        tar_path (str): Directory containing tar files.
        max_files (int, optional): Maximum number of tar files to process.
        
    Returns:
        dict: Collected statistics including:
            - pr_states: Counter of PR states
            - pr_body_lengths: List of PR body lengths
            - repo_languages: Counter of repository languages
            - additions: List of addition counts
            - deletions: List of deletion counts
            - changed_files: List of changed file counts
            - commits: List of commit counts
            - merged_ratio: Ratio of merged PRs
            - total_prs: Total number of PRs processed
            - errors: List of error messages
    """
    print(f"Finding all tar files in {tar_path}...")
    tar_files = find_all_files(tar_path, extension='.tar')
    print(f"Found {len(tar_files)} tar files")
    
    if max_files is not None:
        tar_files = tar_files[:max_files]
        print(f"Limiting to first {max_files} files")
    
    stats = {
        'pr_states': Counter(),
        'pr_body_lengths': [],
        'repo_languages': Counter(),
        'additions': [],
        'deletions': [],
        'changed_files': [],
        'commits': [],
        'merged_count': 0,
        'total_prs': 0,
        'issuecomment_counts': [],  # Number of issuecomments per PR
        'total_issuecomments': 0,  # Total number of issuecomment files
        'errors': []
    }
    
    print(f"\nCollecting PR statistics from tar files...")
    
    for tar_file in tqdm(tar_files, desc="Processing"):
        try:
            tar, files_dict = read_all_files_in_tar(tar_file, extension='any')
            
            # Count issuecomments per PR
            # First, group issuecomments by PR
            pr_issuecomment_map = defaultdict(int)
            
            # Find PR JSON files and count their issuecomments
            pr_files = {}
            for filename, member in files_dict.items():
                if filename.endswith('.json') and '#pull#' in filename and not filename.endswith('_swebench.json') and not '#issuecomment-' in filename:
                    pr_files[filename] = member
                elif '#issuecomment-' in filename and filename.endswith('.json'):
                    # Extract PR identifier from issuecomment filename
                    # Format: repo#owner#pull#number#issuecomment-id.json
                    parts = filename.split('#issuecomment-')
                    if len(parts) > 0:
                        pr_key = parts[0]  # Everything before #issuecomment-
                        pr_issuecomment_map[pr_key] += 1
                        stats['total_issuecomments'] += 1
            
            # Process PR files
            for filename, member in pr_files.items():
                stats['total_prs'] += 1
                
                # Get issuecomment count for this PR
                pr_key = filename.rsplit('.json', 1)[0]  # Remove .json extension
                issuecomment_count = pr_issuecomment_map.get(pr_key, 0)
                stats['issuecomment_counts'].append(issuecomment_count)
                
                try:
                    # Read and parse JSON
                    json_data = tar.extractfile(member).read()
                    pr_data = json.loads(json_data)
                    
                    # Collect PR state
                    state = pr_data.get('state', 'unknown')
                    stats['pr_states'][state] += 1
                    
                    # Check if merged
                    if pr_data.get('merged', False) or pr_data.get('merged_at') is not None:
                        stats['merged_count'] += 1
                    
                    # Collect PR body length
                    body = pr_data.get('body', '')
                    if body:
                        stats['pr_body_lengths'].append(len(body))
                    
                    # Collect repository language
                    repo_info = pr_data.get('head', {}).get('repo') or pr_data.get('base', {}).get('repo') or {}
                    language = repo_info.get('language', 'unknown')
                    if language:
                        stats['repo_languages'][language] += 1
                    
                    # Collect code change information
                    additions = pr_data.get('additions')
                    if additions is not None:
                        stats['additions'].append(additions)
                    
                    deletions = pr_data.get('deletions')
                    if deletions is not None:
                        stats['deletions'].append(deletions)
                    
                    changed_files = pr_data.get('changed_files')
                    if changed_files is not None:
                        stats['changed_files'].append(changed_files)
                    
                    commits = pr_data.get('commits')
                    if commits is not None:
                        stats['commits'].append(commits)
                
                except json.JSONDecodeError as e:
                    stats['errors'].append(f"{os.path.basename(tar_file)}/{filename}: JSON decode error")
                except Exception as e:
                    stats['errors'].append(f"{os.path.basename(tar_file)}/{filename}: {str(e)}")
            
            tar.close()
        except Exception as e:
            stats['errors'].append(f"{os.path.basename(tar_file)}: {str(e)}")
    
    return stats


def create_statistics_visualization(stats: dict, output_file: str = 'swebench_statistics.png') -> None:
    """Create visualization charts for PR statistics.
    
    Args:
        stats (dict): Statistics dictionary from collect_pr_statistics.
        output_file (str): Output file path for the visualization.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return
    
    # Create figure with gridspec: left side 3x3 charts, right side statistics text
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle('SWE-Bench Dataset Statistics Overview', fontsize=16, fontweight='bold', y=0.995)
    
    # Create gridspec: 3 rows, 4 columns (3 for charts, 1 for stats)
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.6], hspace=0.3, wspace=0.4)
    
    # 1. PR State Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    if stats['pr_states']:
        states = list(stats['pr_states'].keys())
        counts = [stats['pr_states'][s] for s in states]
        bars = ax1.bar(states, counts, alpha=0.7, color=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax1.set_title('PR State Distribution', fontweight='bold')
        ax1.set_xlabel('State')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3, axis='y')
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100 if total > 0 else 0
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No PR state data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('PR State Distribution', fontweight='bold')
    
    # 2. Repository Language Distribution (Bar Chart - Top 10)
    ax2 = fig.add_subplot(gs[0, 1])
    if stats['repo_languages']:
        languages_data = stats['repo_languages'].most_common(10)
        languages = [lang[0] for lang in languages_data]
        counts = [lang[1] for lang in languages_data]
        bars = ax2.barh(languages, counts, alpha=0.7, color=plt.cm.Set3(range(len(languages))))
        ax2.set_title('Repository Language Distribution (Top 10)', fontweight='bold')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Language')
        ax2.grid(True, alpha=0.3, axis='x')
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax2.text(count, bar.get_y() + bar.get_height()/2, 
                   f' {count}', ha='left', va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No language data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Repository Language Distribution', fontweight='bold')
    
    # 3. PR Body Length Distribution (Histogram)
    ax3 = fig.add_subplot(gs[0, 2])
    if stats['pr_body_lengths']:
        lengths = stats['pr_body_lengths']
        ax3.hist(lengths, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax3.set_title('PR Body Length Distribution', fontweight='bold')
        ax3.set_xlabel('Body Length (characters)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
        # Add statistics text
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        ax3.axvline(mean_len, color='r', linestyle='--', label=f'Mean: {mean_len:.0f}')
        ax3.axvline(median_len, color='g', linestyle='--', label=f'Median: {median_len:.0f}')
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No body length data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('PR Body Length Distribution', fontweight='bold')
    
    # Helper function to filter outliers using percentile
    def filter_outliers(data, percentile=95):
        """Filter outliers using percentile method."""
        if not data:
            return data, 0, 0
        threshold = np.percentile(data, percentile)
        filtered = [x for x in data if x <= threshold]
        outliers_count = len(data) - len(filtered)
        return filtered, threshold, outliers_count
    
    # 4. Code Additions Distribution (Histogram with outlier filtering)
    ax4 = fig.add_subplot(gs[1, 0])
    if stats['additions']:
        additions = stats['additions']
        additions_filtered, threshold, outliers = filter_outliers(additions, percentile=95)
        
        # Plot filtered data
        ax4.hist(additions_filtered, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        ax4.set_title(f'Code Additions Distribution\n(95th percentile: ≤{threshold:.0f}, {outliers} outliers)', 
                     fontweight='bold', fontsize=10)
        ax4.set_xlabel('Lines Added')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(additions_filtered)
        median_val = np.median(additions_filtered)
        ax4.axvline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.0f}')
        ax4.axvline(median_val, color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.0f}')
        ax4.legend(fontsize=7, loc='upper right')
        
        # Add note about max value
        if outliers > 0:
            max_val = max(additions)
            ax4.text(0.98, 0.02, f'Max: {max_val:,}', transform=ax4.transAxes, 
                    ha='right', va='bottom', fontsize=8, style='italic')
    else:
        ax4.text(0.5, 0.5, 'No additions data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Code Additions Distribution', fontweight='bold')
    
    # 5. Code Deletions Distribution (Histogram with outlier filtering)
    ax5 = fig.add_subplot(gs[1, 1])
    if stats['deletions']:
        deletions = stats['deletions']
        deletions_filtered, threshold, outliers = filter_outliers(deletions, percentile=95)
        
        # Plot filtered data
        ax5.hist(deletions_filtered, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax5.set_title(f'Code Deletions Distribution\n(95th percentile: ≤{threshold:.0f}, {outliers} outliers)', 
                     fontweight='bold', fontsize=10)
        ax5.set_xlabel('Lines Deleted')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(deletions_filtered)
        median_val = np.median(deletions_filtered)
        ax5.axvline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.0f}')
        ax5.axvline(median_val, color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.0f}')
        ax5.legend(fontsize=7, loc='upper right')
        
        # Add note about max value
        if outliers > 0:
            max_val = max(deletions)
            ax5.text(0.98, 0.02, f'Max: {max_val:,}', transform=ax5.transAxes, 
                    ha='right', va='bottom', fontsize=8, style='italic')
    else:
        ax5.text(0.5, 0.5, 'No deletions data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Code Deletions Distribution', fontweight='bold')
    
    # 6. Changed Files Distribution (Histogram with outlier filtering)
    ax6 = fig.add_subplot(gs[1, 2])
    if stats['changed_files']:
        changed = stats['changed_files']
        changed_filtered, threshold, outliers = filter_outliers(changed, percentile=95)
        
        # Plot filtered data
        ax6.hist(changed_filtered, bins=30, alpha=0.7, color='#f39c12', edgecolor='black')
        ax6.set_title(f'Changed Files Distribution\n(95th percentile: ≤{threshold:.0f}, {outliers} outliers)', 
                     fontweight='bold', fontsize=10)
        ax6.set_xlabel('Number of Files Changed')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(changed_filtered)
        median_val = np.median(changed_filtered)
        ax6.axvline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
        ax6.axvline(median_val, color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
        ax6.legend(fontsize=7, loc='upper right')
        
        # Add note about max value
        if outliers > 0:
            max_val = max(changed)
            ax6.text(0.98, 0.02, f'Max: {max_val}', transform=ax6.transAxes, 
                    ha='right', va='bottom', fontsize=8, style='italic')
    else:
        ax6.text(0.5, 0.5, 'No changed files data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Changed Files Distribution', fontweight='bold')
    
    # 7. Commits Distribution (Histogram with outlier filtering)
    ax7 = fig.add_subplot(gs[2, 0])
    if stats['commits']:
        commits = stats['commits']
        commits_filtered, threshold, outliers = filter_outliers(commits, percentile=95)
        
        # Plot filtered data
        ax7.hist(commits_filtered, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax7.set_title(f'Commits per PR Distribution\n(95th percentile: ≤{threshold:.0f}, {outliers} outliers)', 
                     fontweight='bold', fontsize=10)
        ax7.set_xlabel('Number of Commits')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(commits_filtered)
        median_val = np.median(commits_filtered)
        ax7.axvline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
        ax7.axvline(median_val, color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
        ax7.legend(fontsize=7, loc='upper right')
        
        # Add note about max value
        if outliers > 0:
            max_val = max(commits)
            ax7.text(0.98, 0.02, f'Max: {max_val}', transform=ax7.transAxes, 
                    ha='right', va='bottom', fontsize=8, style='italic')
    else:
        ax7.text(0.5, 0.5, 'No commits data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Commits per PR Distribution', fontweight='bold')
    
    # 8. Additions vs Deletions Scatter Plot (filtered)
    ax8 = fig.add_subplot(gs[2, 1])
    if stats['additions'] and stats['deletions']:
        additions = stats['additions']
        deletions = stats['deletions']
        # Align lengths
        min_len = min(len(additions), len(deletions))
        additions_aligned = additions[:min_len]
        deletions_aligned = deletions[:min_len]
        
        # Filter outliers for both dimensions
        additions_p95 = np.percentile(additions_aligned, 95)
        deletions_p95 = np.percentile(deletions_aligned, 95)
        
        # Keep points within 95th percentile for both dimensions
        filtered_pairs = [(a, d) for a, d in zip(additions_aligned, deletions_aligned) 
                         if a <= additions_p95 and d <= deletions_p95]
        
        if filtered_pairs:
            filtered_add, filtered_del = zip(*filtered_pairs)
            ax8.scatter(filtered_add, filtered_del, alpha=0.5, s=20, color='#3498db')
            ax8.set_title(f'Additions vs Deletions\n(Filtered to 95th percentile)', 
                         fontweight='bold', fontsize=10)
            ax8.set_xlabel('Lines Added')
            ax8.set_ylabel('Lines Deleted')
            ax8.grid(True, alpha=0.3)
            
            # Add note about outliers
            outliers_count = len(additions_aligned) - len(filtered_add)
            if outliers_count > 0:
                ax8.text(0.02, 0.98, f'{outliers_count} outliers excluded', 
                        transform=ax8.transAxes, ha='left', va='top', 
                        fontsize=8, style='italic',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax8.text(0.5, 0.5, 'No data after filtering', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Additions vs Deletions', fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'No additions/deletions data', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Additions vs Deletions', fontweight='bold')
    
    # 9. IssueComment Count Distribution
    ax9 = fig.add_subplot(gs[2, 2])
    if stats['issuecomment_counts']:
        comment_counts = stats['issuecomment_counts']
        comment_counts_filtered, threshold, outliers = filter_outliers(comment_counts, percentile=95)
        
        # Plot filtered data
        ax9.hist(comment_counts_filtered, bins=30, alpha=0.7, color='#16a085', edgecolor='black')
        ax9.set_title(f'IssueComment Count per PR\n(95th percentile: ≤{threshold:.0f}, {outliers} outliers)', 
                     fontweight='bold', fontsize=10)
        ax9.set_xlabel('Number of IssueComments')
        ax9.set_ylabel('Frequency')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(comment_counts_filtered)
        median_val = np.median(comment_counts_filtered)
        ax9.axvline(mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
        ax9.axvline(median_val, color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
        ax9.legend(fontsize=7, loc='upper right')
        
        # Add note about max value and total
        ax9.text(0.98, 0.02, f'Max: {max(comment_counts)}\nTotal: {stats["total_issuecomments"]:,}', 
                transform=ax9.transAxes, ha='right', va='bottom', fontsize=8, style='italic')
    else:
        ax9.text(0.5, 0.5, 'No issuecomment data', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('IssueComment Count per PR', fontweight='bold')
    
    # 10. Summary Statistics Text (right side, spanning all 3 rows)
    ax10 = fig.add_subplot(gs[:, 3])
    ax10.axis('off')
    
    summary_text = []
    summary_text.append("SUMMARY STATISTICS")
    summary_text.append("=" * 30)
    summary_text.append(f"Total PRs: {stats['total_prs']:,}")
    
    if stats['pr_states']:
        summary_text.append(f"\nPR States:")
        for state, count in stats['pr_states'].most_common():
            summary_text.append(f"  {state}: {count:,} ({count/stats['total_prs']*100:.1f}%)")
    
    if stats['total_prs'] > 0:
        summary_text.append(f"\nMerged Ratio: {stats['merged_count']/stats['total_prs']*100:.1f}%")
    
    if stats['additions']:
        summary_text.append(f"\nAdditions:")
        summary_text.append(f"  Mean: {np.mean(stats['additions']):.1f}")
        summary_text.append(f"  Median: {np.median(stats['additions']):.1f}")
        summary_text.append(f"  Max: {max(stats['additions']):,}")
    
    if stats['deletions']:
        summary_text.append(f"\nDeletions:")
        summary_text.append(f"  Mean: {np.mean(stats['deletions']):.1f}")
        summary_text.append(f"  Median: {np.median(stats['deletions']):.1f}")
        summary_text.append(f"  Max: {max(stats['deletions']):,}")
    
    if stats['changed_files']:
        summary_text.append(f"\nChanged Files:")
        summary_text.append(f"  Mean: {np.mean(stats['changed_files']):.1f}")
        summary_text.append(f"  Median: {np.median(stats['changed_files']):.1f}")
        summary_text.append(f"  Max: {max(stats['changed_files'])}")
    
    if stats['commits']:
        summary_text.append(f"\nCommits:")
        summary_text.append(f"  Mean: {np.mean(stats['commits']):.1f}")
        summary_text.append(f"  Median: {np.median(stats['commits']):.1f}")
        summary_text.append(f"  Max: {max(stats['commits'])}")
    
    if stats['pr_body_lengths']:
        summary_text.append(f"\nBody Length:")
        summary_text.append(f"  Mean: {np.mean(stats['pr_body_lengths']):.0f}")
        summary_text.append(f"  Median: {np.median(stats['pr_body_lengths']):.0f}")
        summary_text.append(f"  Max: {max(stats['pr_body_lengths']):,}")
    
    if stats['issuecomment_counts']:
        summary_text.append(f"\nIssueComments:")
        summary_text.append(f"  Total: {stats['total_issuecomments']:,}")
        summary_text.append(f"  Mean per PR: {np.mean(stats['issuecomment_counts']):.1f}")
        summary_text.append(f"  Median per PR: {np.median(stats['issuecomment_counts']):.1f}")
        summary_text.append(f"  Max per PR: {max(stats['issuecomment_counts'])}")
        zero_comments = sum(1 for c in stats['issuecomment_counts'] if c == 0)
        summary_text.append(f"  PRs with 0 comments: {zero_comments} ({zero_comments/len(stats['issuecomment_counts'])*100:.1f}%)")
    
    if stats['errors']:
        summary_text.append(f"\nErrors: {len(stats['errors'])}")
    
    ax10.text(0.05, 0.98, '\n'.join(summary_text), transform=ax10.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


def print_pr_statistics(stats: dict) -> None:
    """Print detailed PR statistics.
    
    Args:
        stats (dict): Statistics dictionary from collect_pr_statistics.
    """
    print("\n" + "="*80)
    print("PR STATISTICS SUMMARY")
    print("="*80)
    
    print(f"\nTotal PRs processed: {stats['total_prs']:,}")
    
    if stats['pr_states']:
        print(f"\nPR State Distribution:")
        for state, count in stats['pr_states'].most_common():
            percentage = (count / stats['total_prs'] * 100) if stats['total_prs'] > 0 else 0
            print(f"  {state}: {count:,} ({percentage:.1f}%)")
    
    if stats['total_prs'] > 0:
        merged_ratio = stats['merged_count'] / stats['total_prs'] * 100
        print(f"\nMerged PRs: {stats['merged_count']:,} ({merged_ratio:.1f}%)")
    
    if stats['repo_languages']:
        print(f"\nRepository Languages (Top 10):")
        for lang, count in stats['repo_languages'].most_common(10):
            percentage = (count / stats['total_prs'] * 100) if stats['total_prs'] > 0 else 0
            print(f"  {lang}: {count:,} ({percentage:.1f}%)")
    
    if stats['pr_body_lengths']:
        print(f"\nPR Body Length Statistics:")
        print(f"  Count: {len(stats['pr_body_lengths']):,}")
        print(f"  Mean: {np.mean(stats['pr_body_lengths']):.0f} characters")
        print(f"  Median: {np.median(stats['pr_body_lengths']):.0f} characters")
        print(f"  Min: {min(stats['pr_body_lengths']):,} characters")
        print(f"  Max: {max(stats['pr_body_lengths']):,} characters")
    
    if stats['additions']:
        print(f"\nCode Additions Statistics:")
        print(f"  Count: {len(stats['additions']):,}")
        print(f"  Mean: {np.mean(stats['additions']):.1f} lines")
        print(f"  Median: {np.median(stats['additions']):.1f} lines")
        print(f"  Min: {min(stats['additions']):,} lines")
        print(f"  Max: {max(stats['additions']):,} lines")
    
    if stats['deletions']:
        print(f"\nCode Deletions Statistics:")
        print(f"  Count: {len(stats['deletions']):,}")
        print(f"  Mean: {np.mean(stats['deletions']):.1f} lines")
        print(f"  Median: {np.median(stats['deletions']):.1f} lines")
        print(f"  Min: {min(stats['deletions']):,} lines")
        print(f"  Max: {max(stats['deletions']):,} lines")
    
    if stats['changed_files']:
        print(f"\nChanged Files Statistics:")
        print(f"  Count: {len(stats['changed_files']):,}")
        print(f"  Mean: {np.mean(stats['changed_files']):.1f} files")
        print(f"  Median: {np.median(stats['changed_files']):.1f} files")
        print(f"  Min: {min(stats['changed_files'])} files")
        print(f"  Max: {max(stats['changed_files'])} files")
    
    if stats['commits']:
        print(f"\nCommits Statistics:")
        print(f"  Count: {len(stats['commits']):,}")
        print(f"  Mean: {np.mean(stats['commits']):.1f} commits")
        print(f"  Median: {np.median(stats['commits']):.1f} commits")
        print(f"  Min: {min(stats['commits'])} commits")
        print(f"  Max: {max(stats['commits'])} commits")
    
    if stats['issuecomment_counts']:
        print(f"\nIssueComment Statistics:")
        print(f"  Total issuecomment files: {stats['total_issuecomments']:,}")
        print(f"  PRs with data: {len(stats['issuecomment_counts']):,}")
        print(f"  Mean comments per PR: {np.mean(stats['issuecomment_counts']):.1f}")
        print(f"  Median comments per PR: {np.median(stats['issuecomment_counts']):.1f}")
        print(f"  Min: {min(stats['issuecomment_counts'])} comments")
        print(f"  Max: {max(stats['issuecomment_counts'])} comments")
        zero_comments = sum(1 for c in stats['issuecomment_counts'] if c == 0)
        print(f"  PRs with 0 comments: {zero_comments:,} ({zero_comments/len(stats['issuecomment_counts'])*100:.1f}%)")
        one_or_more = len(stats['issuecomment_counts']) - zero_comments
        print(f"  PRs with 1+ comments: {one_or_more:,} ({one_or_more/len(stats['issuecomment_counts'])*100:.1f}%)")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        if len(stats['errors']) <= 10:
            for error in stats['errors']:
                print(f"  - {error}")
        else:
            print("  (showing first 10 errors)")
            for error in stats['errors'][:10]:
                print(f"  - {error}")
    
    print("="*80)


if __name__ == '__main__':
    # main()

    # 统计所有tar中oracle_files为空列表的比例
    # check_oracle_files_empty_ratio(tar_path, max_files=None)
    
    # 收集PR统计信息并生成可视化
    print("Collecting PR statistics...")
    pr_stats = collect_pr_statistics(tar_path, max_files=None)
    
    print_pr_statistics(pr_stats)
    
    if HAS_MATPLOTLIB:
        print("\nGenerating visualization...")
        create_statistics_visualization(pr_stats, output_file='swebench_statistics.png')
    else:
        print("\nSkipping visualization (matplotlib not available)")