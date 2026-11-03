"""Extract the exact archived inference library into a new local directory."""
from pathlib import Path
import argparse,zipfile
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
target=a.output.resolve();assert not target.exists(),'choose_a_fresh_runtime_directory'
archive=Path(__file__).resolve().parents[1]/'runtime/ultralytics_runtime_8_4_193.zip'
with zipfile.ZipFile(archive)as z:
 for row in z.infolist():
  assert (target/row.filename).resolve().is_relative_to(target),'unsafe_archive_member'
 target.mkdir(parents=True);z.extractall(target)
print('Runtime extracted. Add this directory to PYTHONPATH before importing Ultralytics: '+str(target))
print('No GPU import, model inference, network or hardware control performed.')
