
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import pathlib
import threading
import urllib.parse
from queue import Queue, Empty

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

BASE = "https://huggingface.co"
DOCS_ROOT = f"{BASE}/docs"

# A reasonable default set of sub-sections under /docs to seed.
DEFAULT_INCLUDES = [
    "transformers",
    "datasets",
    "diffusers",
    "tokenizers",
    "accelerate",
    "peft",
    "trl",
    "optimum",
    "evaluate",
    "hub",
    "hub/python",
    "inference-endpoints",
    "text-generation-inference",
    "text-embeddings-inference",
    "lighteval",
    "tasks",
    "autosubmit",
    "gradio",
    "smolagents",
    "lerobot",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en;q=0.8, *;q=0.5",
}

def norm_url(url: str) -> str:
    # """Normalize URL to absolute https and strip fragment."""
    if not url:
        return ""
    url = urllib.parse.urljoin(DOCS_ROOT + "/", url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    if parsed.netloc != urllib.parse.urlparse(BASE).netloc:
        return ""
    # drop fragment and normalize path
    cleaned = parsed._replace(fragment="").geturl()
    return cleaned

def is_docs_url(url: str) -> bool:
    return url.startswith(DOCS_ROOT)

def filter_includes(url: str, includes: list[str]) -> bool:
    if not includes:
        return True
    rel = url[len(DOCS_ROOT):].lstrip("/")
    return any(rel.startswith(p.strip("/")) for p in includes)

def filter_langs(url: str, langs: set[str] | None) -> bool:
    if not langs:
        return True
    # detect a segment that looks like 'en', 'zh', 'fr', 'pt-br', etc.
    parts = urllib.parse.urlparse(url).path.split("/")
    for seg in parts:
        if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", seg or ""):
            base = seg.split("-")[0]
            return base in langs
    return True  # allow pages without explicit lang segment

def filter_versions(url: str, only_main: bool = True) -> bool:
    """
    过滤版本号URL，只保留最新（main）版本的文档
    
    版本号格式示例：
    - https://huggingface.co/docs/transformers/v4.57.0/...
    - https://huggingface.co/docs/transformers/v4.57.1/...
    - https://huggingface.co/docs/transformers/main/... (或不带版本号)
    """
    if not only_main:
        return True
    
    path = urllib.parse.urlparse(url).path
    parts = path.split("/")
    
    # 检查路径中是否包含版本号
    # 版本号格式: v + 数字.数字.数字 (如 v4.57.0, v1.2.3)
    for part in parts:
        # 匹配版本号模式
        if re.match(r'^v\d+\.\d+', part):
            # 这是一个版本号路径，排除它
            return False
        # 也排除类似 "en/v4.57.0" 这样的情况
        if re.match(r'^v\d+', part):
            return False
    
    return True  # 不包含版本号，允许通过

def safe_filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    # ensure index.html for trailing slash
    if path.endswith("/"):
        path += "index"
    # strip /docs/ prefix
    if path.startswith("/docs/"):
        path = path[len("/docs/"):]
    path = path.lstrip("/")
    # force .html
    if not path.endswith(".html"):
        path += ".html"
    return path

def save_html(out_dir: pathlib.Path, url: str, html: str):
    fname = safe_filename_from_url(url)
    fpath = out_dir / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)

def file_exists(out_dir: pathlib.Path, url: str) -> bool:
    """检查文件是否已经下载"""
    fname = safe_filename_from_url(url)
    fpath = out_dir / fname
    return fpath.exists() and fpath.stat().st_size > 100  # 至少100字节

def build_robots() -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(f"{BASE}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass
    return rp

def polite_get(session: requests.Session, url: str, timeout: float = 30.0, max_retries: int = 5) -> tuple[int, str | None]:
    """改进的请求函数，更好地处理429错误"""
    backoff = 1.0  # 初始退避时间增加
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
            status = resp.status_code
            
            if status == 200:
                return status, resp.text
            
            if status == 429:
                # 429错误：使用指数退避，并且时间更长
                wait_time = backoff * (2 ** attempt)
                print(f"[!] Rate limit (429) for {url}, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait_time)
                continue
            
            if status == 503:
                # 服务不可用，稍微等待
                wait_time = backoff * 2
                print(f"[!] Service unavailable (503) for {url}, waiting {wait_time:.1f}s", flush=True)
                time.sleep(wait_time)
                continue
            
            # 其他错误状态码
            return status, None
            
        except requests.Timeout:
            print(f"[!] Timeout for {url} (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(backoff)
            backoff *= 2
        except requests.RequestException as e:
            print(f"[!] Request error for {url}: {e}", flush=True)
            time.sleep(backoff)
            backoff *= 2
    
    return 0, None

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        u = norm_url(urllib.parse.urljoin(base_url, a["href"]))  # resolve relative
        if u:
            urls.append(u)
    # canonical links (if any)
    for l in soup.find_all("link", href=True):
        u = norm_url(urllib.parse.urljoin(base_url, l["href"]))
        if u:
            urls.append(u)
    # also consider next/prev in navs
    for rel in ("next", "prev"):
        for l in soup.find_all("link", rel=lambda v: v and rel in v):
            href = l.get("href")
            u = norm_url(urllib.parse.urljoin(base_url, href))
            if u:
                urls.append(u)
    return urls

def crawl(out_dir: str,
          includes: list[str],
          langs: set[str] | None,
          max_pages: int,
          concurrency: int,
          delay: float,
          resume: bool = True,
          only_main_version: bool = True):
    
    print(f"[info] Starting crawl...", flush=True)
    print(f"[info] Output dir: {out_dir}", flush=True)
    print(f"[info] Resume mode: {resume}", flush=True)
    print(f"[info] Only main version: {only_main_version}", flush=True)
    
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rp = build_robots()
    if not rp.can_fetch("*", DOCS_ROOT):
        print("[!] robots.txt disallows /docs . Aborting.")
        return 1

    session = requests.Session()
    seen: set[str] = set()
    q: Queue[str] = Queue()

    # Seeds: docs index + selected sub-sections
    seeds = [DOCS_ROOT] + [f"{DOCS_ROOT}/{p.strip('/')}" for p in includes]
    for s in seeds:
        seed_url = s if s.endswith("/") else s + "/"
        q.put(seed_url)

    # thread-safe counters/logging
    lock = threading.Lock()
    count = 0
    skip_count = 0
    fail_count = 0
    rate_limit_count = 0
    stop = threading.Event()

    def worker():
        nonlocal count, skip_count, fail_count, rate_limit_count
        
        while not stop.is_set():
            try:
                url = q.get(timeout=1.0)  # 增加超时时间
            except Empty:
                continue
            
            try:
                # 检查是否达到最大页数
                if max_pages and count >= max_pages:
                    stop.set()
                    continue

                # 检查是否已经处理过
                if url in seen:
                    continue
                    
                # URL过滤
                if not is_docs_url(url):
                    continue
                if not filter_includes(url, includes):
                    continue
                if not filter_langs(url, langs):
                    continue
                if not filter_versions(url, only_main_version):
                    continue

                # 标记为已见
                with lock:
                    seen.add(url)
                
                # 断点续传：检查文件是否已存在
                if resume and file_exists(out_path, url):
                    with lock:
                        skip_count += 1
                        if skip_count % 10 == 0:
                            print(f"[info] skipped {skip_count} existing files", flush=True)
                    
                    # 即使文件存在，也要提取链接（确保完整爬取）
                    try:
                        fname = safe_filename_from_url(url)
                        fpath = out_path / fname
                        with open(fpath, 'r', encoding='utf-8') as f:
                            html = f.read()
                        for link in extract_links(html, url):
                            if link.startswith(DOCS_ROOT) and link not in seen:
                                q.put(link)
                    except:
                        pass
                    continue

                # 获取页面
                status, html = polite_get(session, url)
                
                if status == 200 and html:
                    save_html(out_path, url, html)
                    with lock:
                        count += 1
                        if count % 10 == 0:
                            print(f"[info] saved {count} pages | skipped {skip_count} | failed {fail_count} | rate_limited {rate_limit_count}", flush=True)
                    
                    # 发现新链接
                    for link in extract_links(html, url):
                        if link.startswith(DOCS_ROOT) and link not in seen:
                            q.put(link)
                elif status == 429:
                    with lock:
                        rate_limit_count += 1
                    # 429错误：重新加入队列，稍后重试
                    q.put(url)
                    with lock:
                        seen.discard(url)  # 从seen中移除，允许重试
                    time.sleep(5.0)  # 全局等待
                else:
                    with lock:
                        fail_count += 1
                
                # 延迟
                time.sleep(delay)
                
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()

    try:
        q.join()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user", flush=True)
        stop.set()

    for t in threads:
        t.join(timeout=2.0)

    print(f"\n[done] Downloaded {count} new pages, skipped {skip_count} existing, {fail_count} failed, {rate_limit_count} rate limited")
    print(f"[done] Total files in {out_dir}: {len(list(out_path.rglob('*.html')))}")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Mirror Hugging Face docs under https://huggingface.co/docs (HTML)" )
    ap.add_argument("--out", required=True, help="Output directory to save HTML files")
    ap.add_argument("--includes", nargs="*", default=DEFAULT_INCLUDES, help="Limit to these subpaths under /docs. Empty = all under /docs" )
    ap.add_argument("--langs", nargs="*", default=["en"], help="Language codes to include (e.g., en zh fr). Empty = all languages" )
    ap.add_argument("--max-pages", type=int, default=1500, help="Max NEW pages to download (0 = no limit)" )
    ap.add_argument("--concurrency", type=int, default=2, help="Number of worker threads (recommend 1-2 to avoid rate limits)" )
    ap.add_argument("--delay", type=float, default=1.0, help="Delay (seconds) between requests per thread (recommend >= 1.0)" )
    ap.add_argument("--no-resume", action="store_true", help="Disable resume mode (re-download existing files)")
    ap.add_argument("--all-versions", action="store_true", help="Download all versions (default: only main/latest version)")
    args = ap.parse_args()

    langs = set(args.langs) if args.langs else None
    includes = args.includes or []

    code = crawl(out_dir=args.out,
                 includes=includes,
                 langs=langs,
                 max_pages=max(0, args.max_pages),
                 concurrency=max(1, args.concurrency),
                 delay=max(0.0, args.delay),
                 resume=not args.no_resume,
                 only_main_version=not args.all_versions)
    sys.exit(code)

if __name__ == "__main__":
    sys.exit(main())

