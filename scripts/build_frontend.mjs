import {build} from 'esbuild';
import {readFile,writeFile,readdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
await build({absWorkingDir:root,entryPoints:['frontend/main.jsx'],outfile:'web/app.js',
  bundle:true,minify:true,target:['es2020'],format:'iife',jsx:'automatic',
  define:{'process.env.NODE_ENV':'"production"'},legalComments:'inline'});
const packages=(await readdir(`${root}node_modules`)).filter(name=>name==='react'||name==='react-dom'||name==='scheduler'||name==='internmap'||name==='d3'||name.startsWith('d3-'));
let licenses='Bundled frontend dependency licenses\n\n';
for(const name of packages){
  let license;
  for(const file of ['LICENSE','LICENSE.md']){
    try{license=await readFile(`${root}node_modules/${name}/${file}`,'utf8');break;}catch{}
  }
  if(license)licenses+=`=== ${name} ===\n${license}\n\n`;
}
await writeFile(`${root}web/vendor-licenses.txt`,licenses);
console.log('Built React + D3 frontend (web/app.js). Python startup needs no Node.js.');
