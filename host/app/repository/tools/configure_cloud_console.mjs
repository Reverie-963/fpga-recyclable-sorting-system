// Receive runtime connection secrets via hidden stdin, then protect them in
// the local user's DPAPI store. No credential in argv, source or output.
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {dirname,resolve} from 'node:path';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const config=process.argv[2];if(!config)throw Error('Provide the local protected configuration path');
const raw=await new Promise((resolveInput,reject)=>{
  let value='';const terminal=process.stdin.isTTY;
  const finish=error=>{process.stdin.removeListener('data',data);process.stdin.removeListener('end',end);if(terminal)process.stdin.setRawMode(false);process.stdin.pause();error?reject(error):resolveInput(value);};
  const data=chunk=>{value+=chunk;if(value.includes('\u0003'))finish(Error('Cancelled'));else if(value.length>65536)finish(Error('Input too large'));else if(/[\r\n]/.test(value))finish();};
  const end=()=>finish();if(terminal)process.stdin.setRawMode(true);process.stderr.write('Ready for device configuration JSON on stdin (input is hidden).\n');
  process.stdin.setEncoding('utf8');process.stdin.on('data',data);process.stdin.once('end',end);process.stdin.resume();
});
let value;try{value=JSON.parse(raw);}catch{throw Error('Provide one configuration JSON object');}
const python=process.platform==='win32'?resolve(root,'.venv/Scripts/python.exe'):resolve(root,'.venv/bin/python');
const child=spawn(python,[resolve(root,'tools/cloud_console_bridge.py'),'--init-config','--config',resolve(config)],{stdio:['pipe','pipe','pipe'],windowsHide:true});
child.stdin.end(JSON.stringify(value));child.stdout.pipe(process.stdout);child.stderr.pipe(process.stderr);
child.on('close',code=>{process.exitCode=code??1;});
