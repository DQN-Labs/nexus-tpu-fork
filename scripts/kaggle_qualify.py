#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Push the Qwen4Exp TPU qualification notebook to Kaggle and fetch results.

Packages this fork's ``qwen4_exp`` modules + tests as an embedded zip (no
GitHub dependency), writes a 6-cell qualification notebook (setup, TPU gate,
pytest, on-TPU execution proof, prefill->decode consistency), saves it with
``SaveAndRunAll`` on a TPU-enabled Kaggle notebook, polls for completion, and
downloads ``qwen4exp_tpu_results.json`` + ``pytest_summary.json``.

Auth: Bearer token from ``opencode.json`` (``mcp.kaggle.headers``) or the
``KAGGLE_MCP_TOKEN`` env var (``Bearer ...`` value, never printed).

Usage:
    python3 scripts/kaggle_qualify.py --push
    python3 scripts/kaggle_qualify.py --poll
    python3 scripts/kaggle_qualify.py --fetch [--out DIR]
    python3 scripts/kaggle_qualify.py --all   # push, wait, fetch
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
PKG = FORK / 'tpu_inference' / 'models' / 'jax' / 'qwen4_exp'
TEST = FORK / 'tests' / 'models' / 'jax' / 'test_qwen4_exp.py'
BASE = 'https://www.kaggle.com/mcp'
KERNEL_ID = 132221189
KERNEL_SLUG = 'nexus-tpu-qwen3-8flashnext'
KERNEL_USER = 'hemanthvattikuti'


def get_token():
    env = __import__('os').environ.get('KAGGLE_MCP_TOKEN')
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


def build_blob():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PKG.glob('*.py')):
            z.write(p, f'tpu_inference/models/jax/qwen4_exp/{p.name}')
        z.write(TEST, 'tests/models/jax/test_qwen4_exp.py')
    return base64.b64encode(buf.getvalue()).decode()


def code_cell(src):
    return {'cell_type': 'code', 'source': src, 'metadata': {},
            'outputs': [], 'execution_count': None}


def md_cell(src):
    return {'cell_type': 'markdown', 'source': src, 'metadata': {}}


def build_notebook():
    blob = build_blob()
    setup = open(Path(__file__).parent / 'kaggle_cells' / 'setup.py').read()
    setup = setup.replace('__BLOB__', blob)
    cells_dir = Path(__file__).parent / 'kaggle_cells'
    cells = [md_cell((cells_dir / 'header.md').read_text()),
             code_cell(setup),
             code_cell((cells_dir / 'tpu_gate.py').read_text()),
             code_cell((cells_dir / 'run_pytest.py').read_text()),
             code_cell((cells_dir / 'tpu_proof.py').read_text()),
             code_cell((cells_dir / 'chunked.py').read_text())]
    return json.dumps({
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.13'},
            'accelerator': 'TPU'},
        'nbformat_minor': 4, 'nbformat': 4, 'cells': cells})


def cmd_push(_args):
    text = build_notebook()
    print(f'notebook chars: {len(text)}')
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
           'kernelExecutionTypeNullable': 'SaveAndRunAll'}
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
    while time.time() < deadline:
        files = list_files()
        if 'qwen4exp_tpu_results.json' in files:
            print('results present')
            return
        print('waiting for run ...')
        time.sleep(120)
    sys.exit('timed out waiting for qwen4exp_tpu_results.json')


def cmd_fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()
    for name in ('qwen4exp_tpu_results.json', 'pytest_summary.json'):
        if name not in files:
            print(f'missing: {name}')
            continue
        with urllib.request.urlopen(files[name], timeout=120) as r:
            (out / name).write_text(r.read().decode())
            print(f'wrote {out / name}')
    res = out / 'qwen4exp_tpu_results.json'
    if res.exists():
        print(res.read_text()[:3000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('push')
    p.set_defaults(fn=cmd_push)
    p = sub.add_parser('poll')
    p.add_argument('--timeout', type=int, default=5400)
    p.set_defaults(fn=cmd_poll)
    p = sub.add_parser('fetch')
    p.add_argument('--out', default='/tmp/qwen4exp_tpu')
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser('all')
    p.add_argument('--timeout', type=int, default=5400)
    p.add_argument('--out', default='/tmp/qwen4exp_tpu')

    def _all(a):
        cmd_push(a)
        cmd_poll(a)
        cmd_fetch(a)
    p.set_defaults(fn=_all)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
