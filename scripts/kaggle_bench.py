#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Push the Qwen3-8B serving-throughput sweep to the Kaggle TPU notebook.

Installs tpu-inference (JAX backend), downloads ~8B weights (Qwen3-8B, or an
ungated fallback), serves with TP=8, and measures TTFT / prefill tok/s /
decode tok/s across context windows plus a batching point.

Usage: python3 scripts/kaggle_bench.py --push | --poll | --fetch | --all
Auth: Bearer token from opencode.json mcp.kaggle.headers, or KAGGLE_MCP_TOKEN.
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

FORK = Path(__file__).resolve().parent.parent
BASE = 'https://www.kaggle.com/mcp'
KERNEL_ID = 132221189
KERNEL_SLUG = 'nexus-tpu-qwen3-8flashnext'
KERNEL_USER = 'hemanthvattikuti'
CELLS = Path(__file__).parent / 'kaggle_cells_bench'


def get_token():
    import os
    env = os.environ.get('KAGGLE_MCP_TOKEN')
    if env:
        return env
    cfg = json.loads((FORK / 'opencode.json').read_text())
    return cfg['mcp']['kaggle']['headers']['authorization']


def rpc(method, params=None, mid=7, timeout=300):
    payload = {'jsonrpc': '2.0', 'method': method}
    if not method.startswith('notifications/'):
        payload['id'] = mid
        payload['params'] = params
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json',
                 'Accept': 'application/json, text/event-stream',
                 'Authorization': get_token()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        sys.exit(f'HTTP {e.code}: {e.read().decode()[:2000]}')
    if not body.strip():
        return {}
    res = json.loads(body.split('data: ', 1)[1]) if not body.startswith('{') \
        else json.loads(body)
    if 'error' in res:
        sys.exit(f'MCP error: {json.dumps(res["error"])[:2000]}')
    return res.get('result', {})


def call_tool(name, arguments):
    rpc('notifications/initialized')
    out = rpc('tools/call', {'name': name, 'arguments': arguments})
    return out['content'][0]['text']


def code_cell(src):
    return {'cell_type': 'code', 'source': src, 'metadata': {},
            'outputs': [], 'execution_count': None}


def md_cell(src):
    return {'cell_type': 'markdown', 'source': src, 'metadata': {}}


def build_notebook(last_cell='sweep.py'):
    if last_cell == 'prod':
        order = ['header_prod.md', 'setup_prod.py', 'resolve_model.py',
                 'serve_prod.py', 'prompts_guarded.py']
    elif last_cell == 'fork':
        order = ['header_fork.md', 'setup_prod.py', 'fork_apply.py',
                 'resolve_model.py', 'inspect_ckpt.py', 'serve_prod.py',
                 'prompts_guarded.py']
    else:
        header = 'header_prompts.md' if last_cell == 'prompts.py' else 'header.md'
        order = [header, 'setup.py', 'weights.py', 'serve.py', last_cell]
    cells = []
    for n in order:
        src = (CELLS / n).read_text()
        cells.append(md_cell(src) if n.endswith('.md') else code_cell(src))
    return json.dumps({
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.13'},
            'accelerator': 'TPU'},
        'nbformat_minor': 4, 'nbformat': 4, 'cells': cells})


def cmd_push(args):
    text = build_notebook(getattr(args, 'last_cell', 'sweep.py'))
    print(f'notebook chars: {len(text)}')
    # Attached data: the NVFP4 export (primary) + GPTQ export (fallback)
    # as Kaggle model sources. Present only for prod runs; harmless
    # otherwise (bench/qualify cells ignore them).
    model_src = ["keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
                 "ram2121/qwen3-8-flash-next-gptq-4bit/transformers/4bit/1"]
    req = {'id': KERNEL_ID, 'hasId': True, 'idNullable': KERNEL_ID,
           'slug': KERNEL_SLUG, 'hasSlug': True, 'slugNullable': KERNEL_SLUG,
           'text': text, 'hasText': True, 'textNullable': text,
           'language': 'python', 'hasLanguage': True,
           'languageNullable': 'python',
           'kernelType': 'notebook', 'hasKernelType': True,
           'kernelTypeNullable': 'notebook',
           'isPrivate': True, 'hasIsPrivate': True, 'isPrivateNullable': True,
           'enableGpu': False, 'hasEnableGpu': True,
           'enableGpuNullable': False,
           'enableTpu': True, 'hasEnableTpu': True, 'enableTpuNullable': True,
           'enableInternet': True, 'hasEnableInternet': True,
           'enableInternetNullable': True,
           'kernelExecutionType': 'SaveAndRunAll',
           'hasKernelExecutionType': True,
           'kernelExecutionTypeNullable': 'SaveAndRunAll',
           'modelDataSources': model_src, 'modelDataSourcesSetter': model_src}
    print(call_tool('save_notebook', {'request': req})[:500])


def list_files():
    txt = call_tool('list_notebook_session_output',
                    {'request': {'userName': KERNEL_USER,
                                 'kernelSlug': KERNEL_SLUG}})
    try:
        return {f['file_name']: f['url'] for f in json.loads(txt)['files']}
    except Exception:
        return {}


def cmd_poll(args):
    deadline = time.time() + args.timeout
    mode = getattr(args, 'last_cell', 'sweep.py')
    targets = ('prompts_results.json',) if mode in ('prompts.py', 'prod', 'fork') \
        else ('sweep_results.json',)
    while time.time() < deadline:
        files = list_files()
        if any(t in files for t in targets):
            print('results present')
            return
        print('waiting for run ...')
        time.sleep(180)
    sys.exit(f'timed out waiting for {targets}')


def cmd_fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()
    for name in ('sweep_results.json', 'prompts_results.json',
                 'weight_choice.json', 'server_info.json', 'model_path.json',
                 'fork_applied.json', 'inspect_results.json'):
        if name not in files:
            print(f'missing: {name}')
            continue
        with urllib.request.urlopen(files[name], timeout=120) as r:
            (out / name).write_text(r.read().decode())
            print(f'wrote {out / name}')
    for res_name in ('sweep_results.json', 'prompts_results.json'):
        res = out / res_name
        if res.exists():
            print(f'--- {res_name} ---')
            print(res.read_text()[:4000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('push')
    p.add_argument('--last-cell', default='sweep.py')
    p.set_defaults(fn=cmd_push)
    p = sub.add_parser('poll')
    p.add_argument('--last-cell', default='sweep.py')
    p.add_argument('--timeout', type=int, default=14400)
    p.set_defaults(fn=cmd_poll)
    p = sub.add_parser('fetch')
    p.add_argument('--out', default='/tmp/qwen3_bench')
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser('all')
    p.add_argument('--timeout', type=int, default=14400)
    p.add_argument('--out', default='/tmp/qwen3_bench')

    def _all(a):
        cmd_push(a)
        cmd_poll(a)
        cmd_fetch(a)
    p.set_defaults(fn=_all)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
