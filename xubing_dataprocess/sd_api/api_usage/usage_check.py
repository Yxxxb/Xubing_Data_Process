#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime as dt
import requests

USAGE_URL = "https://api.openai.com/v1/usage"

def build_headers(api_key: str, org: str = None, project: str = None):
    h = {"Authorization": f"Bearer {api_key}"}
    if org:
        h["OpenAI-Organization"] = org
    if project:
        h["OpenAI-Project"] = project
    return h

def get_range_usage(start_date: str, end_date: str, headers: dict):
    """尝试区间查询；返回 (json, None)；若缺少 date 报错，返回 (None, 'needs_daily')."""
    params = {"start_date": start_date, "end_date": end_date}
    r = requests.get(USAGE_URL, headers=headers, params=params, timeout=30)
    if r.status_code == 400 and "Missing query parameter 'date'" in r.text:
        return None, "needs_daily"
    r.raise_for_status()
    return r.json(), None

def get_day_usage(day: dt.date, headers: dict):
    """单日查询。"""
    r = requests.get(USAGE_URL, headers=headers, params={"date": day.isoformat()}, timeout=30)
    r.raise_for_status()
    return r.json()

def cents_from_payload(payload: dict) -> int:
    """
    尝试从多种结构中提取“总花费（美分）”：
    优先 total_usage；否则汇总 data[*].total_usage / data[*].cost / line_items[*].cost
    """
    if not isinstance(payload, dict):
        return 0
    cents = 0
    v = payload.get("total_usage")
    if isinstance(v, (int, float)):
        cents += int(v)

    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            for key in ("total_usage", "cost"):
                vv = item.get(key)
                if isinstance(vv, (int, float)):
                    cents += int(vv)

    line_items = payload.get("line_items")
    if isinstance(line_items, list):
        for li in line_items:
            vv = li.get("cost")
            if isinstance(vv, (int, float)):
                cents += int(vv)
    return cents

def summarize_range_payload(p: dict):
    """尽力从区间返回里拆出每日（若可）与总计。"""
    total_cents = cents_from_payload(p)
    daily = []
    # 常见：data 里有逐日/逐桶记录
    data = p.get("data")
    if isinstance(data, list):
        for item in data:
            # 尝试取日期/时间戳字段
            day = item.get("date") or item.get("timestamp") or item.get("aggregation_timestamp")
            val = item.get("total_usage") or item.get("cost")
            if day is not None and isinstance(val, (int, float)):
                daily.append((str(day), float(val) / 100.0))
    return total_cents / 100.0, sorted(daily)

def summarize_daily_payloads(day_payloads):
    total_cents = 0
    daily = []
    for d, p in day_payloads:
        cents = cents_from_payload(p)
        total_cents += cents
        daily.append((d.isoformat(), cents / 100.0))
    return total_cents / 100.0, sorted(daily)

def main():
    tz_today = dt.date.today()  # API 按 UTC 记账；这里日期字符串仍按本地生成即可
    default_start = (tz_today - dt.timedelta(days=30)).isoformat()
    default_end = tz_today.isoformat()

    ap = argparse.ArgumentParser(description="Query OpenAI Usage (range + daily fallback)")
    ap.add_argument("--start-date", default=default_start, help="YYYY-MM-DD（默认过去30天）")
    ap.add_argument("--end-date",   default=default_end,   help="YYYY-MM-DD（默认今天）")
    ap.add_argument("--org", help="可选：组织ID")
    ap.add_argument("--project", help="可选：项目ID")
    args = ap.parse_args()

    api_key = "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"
    if not api_key:
        print("请先设置环境变量 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    headers = build_headers(api_key, args.org, args.project)

    # 1) 先试“区间查询”
    payload, hint = get_range_usage(args.start_date, args.end_date, headers)

    daily_breakdown = []
    if hint == "needs_daily":
        # 2) 回退为“按天查询并聚合”
        d0 = dt.date.fromisoformat(args.start_date)
        d1 = dt.date.fromisoformat(args.end_date)
        if d1 < d0:
            print("end_date 早于 start_date", file=sys.stderr)
            sys.exit(3)
        day_payloads = []
        cur = d0
        while cur <= d1:
            dp = get_day_usage(cur, headers)
            day_payloads.append((cur, dp))
            cur += dt.timedelta(days=1)
        total_usd, daily_breakdown = summarize_daily_payloads(day_payloads)
        print(f"[daily mode] Usage from {args.start_date} to {args.end_date}")
        print(f"Total (USD): {total_usd:.4f}")
        if daily_breakdown:
            print("\nDaily breakdown (USD):")
            for d, usd in daily_breakdown:
                print(f"  {d}: {usd:.4f}")
        return

    # 3) 区间查询成功时，尽力汇总
    total_usd, daily_breakdown = summarize_range_payload(payload)
    print(f"[range mode] Usage from {args.start_date} to {args.end_date}")
    if total_usd > 0:
        print(f"Total (USD): {total_usd:.4f}")
    else:
        print("Total not found in standard field; raw payload follows:")
        print(payload)

    if daily_breakdown:
        print("\nDaily breakdown (USD):")
        for d, usd in daily_breakdown:
            print(f"  {d}: {usd:.4f}")

if __name__ == "__main__":
    main()
