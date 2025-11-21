#!/usr/bin/env python3
"""
Generate coding questions from API documentation using GPT-5.
Processes JSONL files efficiently with rate limiting and error handling.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import AsyncOpenAI
from aiofiles import open as aio_open
import logging

# Configuration
API_KEY = "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"
INPUT_DIR = Path("/data/extracted_apis")
OUTPUT_DIR = Path("/data/generated_questions")
PROMPT_FILE = Path("/home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation/prompt_scene.txt")

# Rate limiting settings for Tier 5 (RPM limit: 10,000)
# 极限配置：最大化速度
MAX_CONCURRENT_REQUESTS = 1000  # 极限并发！
REQUESTS_PER_MINUTE = 9500      # 接近10,000 RPM限制
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # 缩短重试延迟
MAX_COMPLETION_TOKENS = 16000  # Increased for gpt-5 reasoning tokens
REQUEST_TIMEOUT = 600.0  # 增加到600秒，给reasoning更多时间
MAX_CONCURRENT_FILES = 30  # 同时处理的文件数（新增！）

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: int, per: int = 60):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default: 60)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until a request can be made."""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0


class QuestionGenerator:
    """Generate questions from API documentation using GPT-5."""
    
    def __init__(self, api_key: str, prompt_template: str):
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.client = AsyncOpenAI(api_key=api_key)
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def build_prompt(self, entry: Dict[str, Any]) -> str:
        """Build prompt from template and entry data."""
        # Extract package name from library field
        package_name = entry.get('library', 'unknown')
        
        # Build content from entry fields
        content_parts = []
        
        # Add basic info
        content_parts.append(f"## API Object: {entry.get('object', 'N/A')}")
        content_parts.append(f"**Kind**: {entry.get('kind', 'N/A')}")
        content_parts.append(f"**Module**: {entry.get('module', 'N/A')}")
        content_parts.append(f"**Signature**: {entry.get('signature', 'N/A')}")
        content_parts.append(f"**Summary**: {entry.get('summary', 'N/A')}")
        
        # Add documentation
        if '[Library Api Doc]' in entry and entry['[Library Api Doc]']:
            content_parts.append("\n## Documentation")
            content_parts.append(entry['[Library Api Doc]'])
        
        # Add code snippet if available
        if '[Code Snippet]' in entry and entry['[Code Snippet]']:
            content_parts.append("\n## Example Code Snippet")
            content_parts.append(entry['[Code Snippet]'])
        
        content = "\n\n".join(content_parts)
        
        # Replace placeholders in template
        prompt = self.prompt_template.replace("{PACKAGE_NAME}", package_name)
        prompt = prompt.replace("{CONTENT}", content)
        
        return prompt
    
    async def call_gpt5(self, prompt: str) -> Optional[str]:
        """Call GPT-5 API with retry logic."""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Wait for rate limiter
                # DISABLED: Let OpenAI handle rate limiting - we have 0 429 errors!
                # await self.rate_limiter.acquire()
                
                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    #model="gpt-4",  # Use gpt-4 as gpt-5 might not be available yet
                    model="gpt-5",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=MAX_COMPLETION_TOKENS,  # Increased for gpt-5 reasoning
                    temperature=1,
                    timeout=REQUEST_TIMEOUT,
                    reasoning_effort="medium",     # 或 "minimal/medium"/"high" 视任务而定
                )
                
                # Debug: Check why response might be empty
                content = response.choices[0].message.content
                if not content or len(content.strip()) == 0:
                    logger.warning(f"Empty content received! finish_reason: {response.choices[0].finish_reason}")
                    if hasattr(response, 'usage'):
                        logger.warning(f"Usage: {response.usage}")
                
                return content
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{RETRY_ATTEMPTS}")
                    await asyncio.sleep(wait_time)
                elif "timeout" in error_msg.lower():
                    logger.error(f"Timeout on attempt {attempt + 1}/{RETRY_ATTEMPTS}")
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(f"API error (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY)
        
        return None
    
    async def process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single API entry to generate a question."""
        async with self.semaphore:
            self.stats['total'] += 1
            
            # Build prompt
            prompt = self.build_prompt(entry)
            
            # Call GPT-5
            response = await self.call_gpt5(prompt)
            
            # Build result - start with the original entry (preserves all original data)
            result = {
                'original_entry': entry,  # Save complete original entry
                'library': entry.get('library'),
                'object': entry.get('object'),
                'kind': entry.get('kind'),
                'module': entry.get('module'),
                'signature': entry.get('signature'),
                'summary': entry.get('summary'),
                'timestamp': datetime.now().isoformat(),
                'success': response is not None and len(response.strip()) > 0 if response else False
            }
            
            if response and len(response.strip()) > 0:
                # BAD_DOCUMENT is also a valid response
                result['question'] = response
                self.stats['success'] += 1
            else:
                if not response:
                    result['error'] = 'API returned None'
                else:
                    result['error'] = 'Empty response from API'
                result['success'] = False
                self.stats['failed'] += 1
                logger.warning(f"Failed to generate question for {entry.get('object')}: {result['error']}")
            
            # Log progress
            if self.stats['total'] % 10 == 0:
                logger.info(f"Progress: {self.stats['total']} processed, "
                          f"{self.stats['success']} success, {self.stats['failed']} failed")
            
            return result
    
    async def process_file(self, input_file: Path, output_file: Path):
        """Process a single JSONL file with streaming write and resume support."""
        logger.info(f"Processing file: {input_file.name}")
        
        # Read all entries
        entries = []
        async with aio_open(input_file, 'r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {input_file.name}: {e}")
        
        total_entries = len(entries)
        logger.info(f"Found {total_entries} entries in {input_file.name}")
        
        # Check if output file exists and count already processed entries
        already_processed = set()
        start_from = 0
        if output_file.exists():
            try:
                async with aio_open(output_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        if line.strip():
                            result = json.loads(line)
                            already_processed.add(result.get('object'))
                start_from = len(already_processed)
                
                # Quick check: if already fully processed, skip
                if start_from >= total_entries:
                    logger.info(f"  ✓ Already completed: {start_from}/{total_entries} entries")
                    return
                
                logger.info(f"  Resuming: {start_from} entries already processed, {total_entries - start_from} remaining")
            except Exception as e:
                logger.warning(f"  Could not read existing file, starting from scratch: {e}")
                start_from = 0
        
        # Prepare output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process entries in batches and write incrementally
        batch_size = MAX_CONCURRENT_REQUESTS
        total_results = start_from
        
        # Open file in append mode if resuming, otherwise write mode
        file_mode = 'a' if start_from > 0 else 'w'
        async with aio_open(output_file, file_mode, encoding='utf-8') as f:
            for i in range(start_from, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                
                # Skip already processed entries
                batch = [e for e in batch if e.get('object') not in already_processed]
                if not batch:
                    continue
                
                # Process this batch concurrently
                tasks = [self.process_entry(entry) for entry in batch]
                results = await asyncio.gather(*tasks)
                
                # Write results immediately
                for result in results:
                    await f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                total_results += len(results)
                logger.info(f"  {input_file.name}: {total_results}/{len(entries)} entries processed")
        
        logger.info(f"Completed {input_file.name}: {total_results} questions generated")
    
    async def process_all_files(self, resume: bool = True):
        """Process all JSONL files in the input directory."""
        # Get all input files
        input_files = sorted(INPUT_DIR.glob("*.jsonl"))
        logger.info(f"Found {len(input_files)} JSONL files to process")
        
        # Note: We don't filter files here anymore
        # Let process_file handle resume logic (it checks which entries are already processed)
        if resume:
            logger.info(f"Resume mode enabled - will skip already processed entries within each file")
        
        if not input_files:
            logger.info("No files to process")
            return
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Pipeline parallel processing with Semaphore (流水线并行!)
        logger.info(f"Processing {len(input_files)} files with {MAX_CONCURRENT_FILES} concurrent files (pipeline mode)")
        
        # Create a semaphore to limit concurrent file processing
        file_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
        
        async def process_with_semaphore(input_file):
            """Process a single file with semaphore control."""
            async with file_semaphore:  # Acquire slot, auto-release when done
                output_file = OUTPUT_DIR / input_file.name
                try:
                    await self.process_file(input_file, output_file)
                except Exception as e:
                    logger.error(f"Failed to process {input_file.name}: {e}")
        
        # Submit all files at once - semaphore controls concurrency
        # Files start processing as soon as a slot is available (no batch waiting!)
        tasks = [process_with_semaphore(f) for f in input_files]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final statistics
        logger.info("="*60)
        logger.info("Processing complete!")
        logger.info(f"Total entries processed: {self.stats['total']}")
        logger.info(f"Successful: {self.stats['success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Success rate: {self.stats['success']/self.stats['total']*100:.2f}%")
        logger.info("="*60)


async def main():
    """Main entry point."""
    # Read prompt template
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Create generator
    generator = QuestionGenerator(API_KEY, prompt_template)
    
    # Process all files
    await generator.process_all_files(resume=True)


if __name__ == "__main__":
    asyncio.run(main())

