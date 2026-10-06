from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

@dataclass
class ProcessStatus:
    running: bool
    pid: int | None = None
    command: list[str] | None = None
    started_at: str | None = None
    exit_code: int | None = None

class BotController:
    """Small local process controller used by the Streamlit UI.

    It controls only paper/research subprocesses. No exchange order API is exposed.
    """
    def __init__(self, root: str | Path = "."):
        self.root=Path(root).resolve(); self.logs=self.root/"logs"; self.logs.mkdir(parents=True,exist_ok=True)
        self.pid_path=self.logs/"ui_paper_bot.pid"; self.meta_path=self.logs/"ui_paper_bot.json"
    def _read_meta(self)->dict:
        try: return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception: return {}
    def _write_meta(self,payload:dict)->None: self.meta_path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    def _pid_alive(self,pid:int)->bool:
        try: os.kill(pid,0); return True
        except PermissionError: return True
        except OSError: return False
    def status(self)->ProcessStatus:
        meta=self._read_meta(); pid=None
        try:
            if self.pid_path.exists(): pid=int(self.pid_path.read_text(encoding="utf-8").strip())
        except Exception: pid=None
        running=bool(pid and self._pid_alive(pid))
        if not running and pid:
            try: self.pid_path.unlink()
            except OSError: pass
        return ProcessStatus(running,pid if running else None,meta.get("command"),meta.get("started_at"),meta.get("exit_code"))
    def start_paper(self,config:str="config.yaml",symbol:str|None=None,timeframe:str|None=None)->ProcessStatus:
        current=self.status()
        if current.running: return current
        cmd=[sys.executable,"-m","ai_trading_lab.main","paper-daemon","--config",config]
        if symbol: cmd += ["--symbol",symbol]
        if timeframe: cmd += ["--timeframe",timeframe]
        env=os.environ.copy(); src=str(self.root/"src"); env["PYTHONPATH"]=src+os.pathsep+env.get("PYTHONPATH","")
        stdout=(self.logs/"ui_paper_bot.log").open("a",encoding="utf-8"); creationflags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
        try: proc=subprocess.Popen(cmd,cwd=self.root,env=env,stdout=stdout,stderr=subprocess.STDOUT,creationflags=creationflags)
        finally: stdout.close()
        from datetime import datetime, timezone
        meta={"command":cmd,"started_at":datetime.now(timezone.utc).isoformat(),"pid":proc.pid}
        self.pid_path.write_text(str(proc.pid),encoding="utf-8"); self._write_meta(meta); return self.status()
    def stop_paper(self)->ProcessStatus:
        current=self.status()
        if not current.running or not current.pid: return current
        pid=current.pid
        if os.name=="nt": subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True,text=True)
        else:
            try: os.kill(pid,signal.SIGTERM)
            except OSError: pass
        try: self.pid_path.unlink()
        except OSError: pass
        meta=self._read_meta(); meta["exit_code"]=0; self._write_meta(meta); return self.status()
    def run_once(self,command:Sequence[str])->tuple[int,str]:
        env=os.environ.copy(); src=str(self.root/"src"); env["PYTHONPATH"]=src+os.pathsep+env.get("PYTHONPATH","")
        proc=subprocess.run(list(command),cwd=self.root,env=env,capture_output=True,text=True,timeout=600)
        output=(proc.stdout or "")+(("\n"+proc.stderr) if proc.stderr else ""); return proc.returncode,output.strip()

def tail_text(path:str|Path,lines:int=80)->str:
    p=Path(path)
    if not p.exists(): return ""
    content=p.read_text(encoding="utf-8",errors="replace").splitlines(); return "\n".join(content[-max(1,lines):])
