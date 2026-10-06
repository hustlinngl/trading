from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Iterable
GROWTH_SCHEMA="""
CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, hypothesis TEXT NOT NULL, config_json TEXT NOT NULL, parent_id TEXT, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL, evaluated_at TEXT NOT NULL, split TEXT, metric TEXT, value REAL, regime TEXT, details_json TEXT);
CREATE INDEX IF NOT EXISTS idx_eval_exp ON evaluations(experiment_id);
CREATE TABLE IF NOT EXISTS model_versions (version_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, role TEXT NOT NULL, parent_version TEXT, status TEXT NOT NULL, score REAL, metrics_json TEXT, artifact_path TEXT);
CREATE TABLE IF NOT EXISTS event_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, provider TEXT, event_time TEXT, feature_json TEXT NOT NULL, horizon INTEGER NOT NULL, realized_return REAL, direction REAL, status TEXT NOT NULL DEFAULT 'open');
CREATE UNIQUE INDEX IF NOT EXISTS ux_event_id ON event_outcomes(event_id);
CREATE INDEX IF NOT EXISTS idx_event_status ON event_outcomes(status);
CREATE TABLE IF NOT EXISTS discoveries (discovery_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, family TEXT, expression TEXT, score REAL, validation_score REAL, complexity INTEGER, status TEXT NOT NULL, evidence_json TEXT);
CREATE TABLE IF NOT EXISTS market_states (state_id TEXT PRIMARY KEY, observed_at TEXT NOT NULL, symbol TEXT NOT NULL, price REAL, payload_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_state_symbol_time ON market_states(symbol, observed_at);
"""
GRAPH_SCHEMA="""
CREATE TABLE IF NOT EXISTS states (state_id TEXT PRIMARY KEY, observed_at TEXT NOT NULL, symbol TEXT NOT NULL, signature TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_states_symbol_time ON states(symbol, observed_at);
CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, observed_at TEXT NOT NULL, state_from TEXT NOT NULL, state_to TEXT NOT NULL, action TEXT NOT NULL, realized_return REAL NOT NULL, hold_bars INTEGER, metadata_json TEXT, transition_version INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS idx_transitions_from ON transitions(state_from);
CREATE INDEX IF NOT EXISTS idx_transitions_action ON transitions(action);
CREATE UNIQUE INDEX IF NOT EXISTS ux_transition_identity ON transitions(symbol, observed_at, state_from, state_to, action);
"""
class ResearchMemoryStore:
    TABLE_COLUMNS={"experiments":"experiment_id,created_at,kind,hypothesis,config_json,parent_id,status","evaluations":"id,experiment_id,evaluated_at,split,metric,value,regime,details_json","model_versions":"version_id,created_at,role,parent_version,status,score,metrics_json,artifact_path","event_outcomes":"id,event_id,provider,event_time,feature_json,horizon,realized_return,direction,status","discoveries":"discovery_id,created_at,family,expression,score,validation_score,complexity,status,evidence_json","market_states":"state_id,observed_at,symbol,price,payload_json","states":"state_id,observed_at,symbol,signature,payload_json","transitions":"id,symbol,observed_at,state_from,state_to,action,realized_return,hold_bars,metadata_json,transition_version"}
    def __init__(self,path:str|Path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.ensure_schema()
    def connect(self): 
        cx=sqlite3.connect(self.path); cx.execute("PRAGMA busy_timeout=5000"); cx.execute("PRAGMA journal_mode=WAL"); cx.execute("PRAGMA synchronous=NORMAL"); return cx
    def ensure_schema(self):
        with self.connect() as cx:
            cx.executescript(GROWTH_SCHEMA); cx.executescript(GRAPH_SCHEMA)
            cols={row[1] for row in cx.execute("PRAGMA table_info(transitions)").fetchall()}
            if "transition_version" not in cols: cx.execute("ALTER TABLE transitions ADD COLUMN transition_version INTEGER NOT NULL DEFAULT 1")
            cx.commit()
    def migrate_legacy(self,paths:Iterable[str|Path])->dict[str,int]:
        counts={}; target=self.path.resolve()
        for raw in paths:
            source=Path(raw)
            if not source.exists() or source.resolve()==target: continue
            alias="legacy"
            with self.connect() as cx:
                try:
                    cx.execute(f"ATTACH DATABASE ? AS {alias}",(str(source),))
                    for table,cols in self.TABLE_COLUMNS.items():
                        if not self._table_exists(cx,f"{alias}.{table}"): continue
                        source_cols={row[1] for row in cx.execute(f"PRAGMA {alias}.table_info({table})").fetchall()}
                        available=[c.strip() for c in cols.split(",") if c.strip() in source_cols]
                        if not available: continue
                        target_cols=",".join(available); before=cx.total_changes
                        cx.execute(f"INSERT OR IGNORE INTO main.{table} ({target_cols}) SELECT {target_cols} FROM {alias}.{table}")
                        counts[table]=counts.get(table,0)+int(cx.total_changes-before)
                    cx.execute(f"DETACH DATABASE {alias}"); cx.commit()
                except sqlite3.DatabaseError:
                    try: cx.execute(f"DETACH DATABASE {alias}")
                    except sqlite3.DatabaseError: pass
        return counts
    @staticmethod
    def _table_exists(cx,qualified_name:str)->bool:
        schema,table=qualified_name.split(".",1) if "." in qualified_name else ("main",qualified_name)
        row=cx.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()
        return row is not None
