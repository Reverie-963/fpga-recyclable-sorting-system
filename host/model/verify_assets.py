from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
expected={
 'host/model/yoloe26_reference.pt':'9d5734b34871aa81412b2fa47176d8f926283bb5adfe3a3f80cc58c5b7332534',
 'host/model/reference_bank.json':'9b6134e32d5c44b3398cdb08c74cdfd915cd46d013d43b9b1ea13920645f5fe8',
 'fpga/build/paired/vision_motion.bit':'054a766d3468282cbfb355dc803b8b9e960749ffde0e80bf0d43e17adebab779',
 'fpga/build/paired/vision_motion.hwh':'b40312875bb53fadec4563109a8508176c75bcf582ce42150f13692e375a5213'}
for name,digest in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(assets=len(expected),passed=True,hardware_access=False)))
