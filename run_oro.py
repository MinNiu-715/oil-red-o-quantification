# -*- coding: utf-8 -*-
"""包装器：以 runpy 方式运行 oro_analyze，绕过直接调用时的静默失败。

脚本查找顺序（按优先级）：
  1. 命令行第一个参数 --script <path>（显式指定）
  2. 环境变量 ORO_SCRIPT
  3. 当前工作目录下的 .workbuddy/tools/oro_analyze.py（项目内副本）
  4. 本 skill 自带副本 scripts/oro_analyze.py（兜底，永远存在）

用法：
  python run_oro.py --input <目录> --out <目录> --mode fixed --thr 40 \
      --min-area 8 --group-from-name
"""
import sys, os, runpy
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

argv = sys.argv[1:]

# 1) --script 显式指定
script = None
if '--script' in argv:
    i = argv.index('--script')
    if i + 1 < len(argv):
        script = Path(argv[i + 1])
        del argv[i:i + 2]

# 2) 环境变量
if script is None and os.environ.get('ORO_SCRIPT'):
    script = Path(os.environ['ORO_SCRIPT'])

# 3) 项目内副本
if script is None:
    cand = Path.cwd() / '.workbuddy' / 'tools' / 'oro_analyze.py'
    if cand.exists():
        script = cand

# 4) 本 skill 自带副本
if script is None or not Path(script).exists():
    script = Path(__file__).resolve().parent / 'oro_analyze.py'

if not Path(script).exists():
    sys.stderr.write(f'[run_oro] 找不到 oro_analyze.py: {script}\n')
    sys.exit(2)

print(f'[run_oro] 使用脚本: {script}', flush=True)
sys.argv = ['oro_analyze.py'] + argv
runpy.run_path(str(script), run_name='__main__')
