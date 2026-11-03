"""Offline package verification. No import of inference or hardware modules."""
from pathlib import Path
import ast,csv,hashlib,json,re,zipfile
root=Path(__file__).resolve().parents[1]
files=[p for p in root.rglob('*')if p.is_file()]
bad=[];python_count=0
secret=re.compile(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
for p in files:
 rel=p.relative_to(root).as_posix()
 assert rel.isascii(),rel
 assert p.suffix.lower() not in ('.pyc','.lic','.pem','.key'),rel
 assert not any(v in p.parts for v in ('.git','.venv','node_modules','__pycache__')),rel
 if p.suffix in ('.py','.json','.jsonl','.md','.txt','.csv','.ps1','.tcl','.html','.js'):
  text=p.read_text(encoding='utf-8-sig')
  if secret.search(text):bad.append(rel)
  if p.suffix=='.py':ast.parse(text,filename=rel);python_count+=1
assert not bad,bad
expected=root/'provenance/file_manifest.json'
checked=0
if expected.exists():
 for row in json.loads(expected.read_text(encoding='utf-8')):
  assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path'];checked+=1
with(root/'metrics/metrics.csv').open(encoding='utf-8-sig',newline='')as f:rows=list(csv.reader(f))
assert rows[0]==['指标名称','类别','数值','单位','测量条件','测试次数 / 时长','证据文件']
links=[]
for row in rows[1:]:
 for ref in row[-1].split(';'):
  target=(root/'metrics'/ref.strip()).resolve()
  assert target.is_relative_to(root) and target.is_file(),ref
  links.append(ref)
for archive in root.rglob('*.zip'):
 with zipfile.ZipFile(archive)as z:assert z.testzip()is None,archive.name
print(json.dumps(dict(passed=True,files=len(files),python_syntax_files=python_count,manifest_hashes_checked=checked,metric_rows=len(rows)-1,metric_evidence_links=len(links),credential_patterns=0,network_or_hardware_access=False)))
