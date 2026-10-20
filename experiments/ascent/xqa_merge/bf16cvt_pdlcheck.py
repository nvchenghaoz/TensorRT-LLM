#!/usr/bin/env python3
# Branch 121 / idea 214: xqa_evidence.py --mode pdlcheck with a reference that can detect the PDL race.
# On main (07a28740) the base kernel_mha already issues its first K/V tile before griddepcontrol.wait
# (XQA_PDL_EARLY_KV, ENABLE_PDL 2 by default), so base-vs-candidate under the PDL writer cannot show the race: both
# arms read token 0 early in the same way (job 965: the token-0 control differed on 0/60 calls). This wrapper runs
# pdlcheck twice with the candidate (cwd tree, default macros) against:
#   ref=wait : the base sha compiled with -DENABLE_PDL=1 (naive PDL: wait before every load, no early K/V);
#              every regular call must be bitwise equal, and the token-0 negative control must differ (race live);
#   ref=base : the base sha with its default macros (early K/V, like the candidate); every regular call must be
#              bitwise equal (its control is informational, since both arms share the early-load schedule).
# Usage: XQA_KV_DTYPE=fp8 python bf16cvt_pdlcheck.py <base_sha> [configs]
import builtins
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'experiments/ascent/xqa_merge'))
import xqa_evidence as xe  # noqa: E402

CWD_TREE = os.path.realpath(os.path.join(os.getcwd(), 'cpp/kernels/xqa'))
CAPTURED = []


def capture_print(*a, **kw):
    text = ' '.join(str(x) for x in a)
    if text.startswith('{'):
        CAPTURED.append(text)
        builtins.print(text[:400] + ' ...', flush=True)
    else:
        builtins.print(*a, **kw)


def run(base_sha, configs, ref_macros):
    orig_compile = xe.compile_xqa

    def compile_xqa(src_dir):
        if os.path.realpath(src_dir) == CWD_TREE:
            return orig_compile(src_dir)
        saved = dict(xe.MACROS)
        xe.MACROS.update(ref_macros)
        try:
            xe.log('reference compile with extra macros %s' % json.dumps(ref_macros))
            return orig_compile(src_dir)
        finally:
            xe.MACROS.clear()
            xe.MACROS.update(saved)

    xe.compile_xqa = compile_xqa
    xe.print = capture_print
    sys.argv = ['xqa_evidence.py', '--mode', 'pdlcheck', '--configs', configs, '--base-commit', base_sha]
    try:
        xe.main()
    finally:
        xe.compile_xqa = orig_compile
    return json.loads(CAPTURED[-1])


def main():
    base_sha = sys.argv[1]
    configs = sys.argv[2] if len(sys.argv) > 2 else '1x47,2x23,3x15,4x11'
    assert xe.KV_DTYPE == 'fp8'
    r_wait = run(base_sha, configs, dict(ENABLE_PDL='1'))
    print('ref=wait (base ENABLE_PDL=1) passed %s, negative control %s' %
          (r_wait['passed'], json.dumps(r_wait['negative_control'])), flush=True)
    r_base = run(base_sha, configs, dict())
    regular_ok = all(v['passed'] for v in r_base['repeated'].values()) and r_base['interleaved']['passed']
    print('ref=base (default macros) regular calls passed %s' % regular_ok, flush=True)
    out = dict(passed=bool(r_wait['passed'] and regular_ok and r_base['readonly_inputs_unchanged']),
               ref_wait=r_wait, ref_base=r_base, ref_base_regular_calls_passed=regular_ok,
               compared_against=('candidate (cwd) under the PDL writer vs base %s compiled (a) with ENABLE_PDL=1 (wait '
                                 'before any load): bitwise every call, and the token-0 negative control must be '
                                 'detected; (b) with default macros (early K/V like the candidate): bitwise every '
                                 'regular call' % base_sha))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
