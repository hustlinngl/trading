from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any
@dataclass(frozen=True)
class EvidenceGate:
    name:str; passed:bool|None; severity:str; detail:str
    def to_dict(self): return asdict(self)
def score_asset_evidence(report:dict[str,Any],*,min_holdout_trades=20,min_profit_factor=1.0,max_drawdown=-0.25,min_dsr=0.80,max_pbo=0.35)->dict[str,Any]:
    gates=[]; q=report.get("quality",{}) or {}; gates.append(EvidenceGate("data_quality",bool(q.get("passed",False)),"hard",str(q.get("reasons",[]))))
    h=report.get("holdout_tuned",{}) or report.get("holdout",{}) or {}; trades=int(h.get("trades",h.get("total_trades",0)) or 0); ret=float(h.get("total_return",h.get("net_compounded_return",0.0)) or 0.0); pf=float(h.get("profit_factor",0.0) or 0.0); dd=float(h.get("max_drawdown",-1.0) or -1.0)
    gates += [EvidenceGate("holdout_trade_support",trades>=int(min_holdout_trades),"hard",f"trades={trades}"),EvidenceGate("holdout_positive_return",ret>0.0,"hard",f"return={ret:.4f}"),EvidenceGate("holdout_profit_factor",pf>=float(min_profit_factor),"hard",f"profit_factor={pf:.3f}"),EvidenceGate("holdout_drawdown",dd>=float(max_drawdown),"hard",f"max_drawdown={dd:.4f}")]
    t=report.get("master_tuning",{}) or {}; st=t.get("stability",{}) or {}; gates.append(EvidenceGate("parameter_stability",bool(st.get("stable",False)),"research",str(st)))
    pl=t.get("negative_control_placebo",{}) or {}; po=pl.get("passes_95pct_negative_control"); gates.append(EvidenceGate("negative_control",None if po is None else bool(po),"research",str(pl)))
    p=t.get("probability_of_backtest_overfitting",{}) or {}; pv=p.get("pbo"); gates.append(EvidenceGate("pbo",None if pv is None else float(pv)<=float(max_pbo),"research",f"pbo={pv}"))
    ev=t.get("final_statistical_evidence",{}) or {}; ds=ev.get("deflated_sharpe_ratio"); gates.append(EvidenceGate("deflated_sharpe",None if ds is None else float(ds)>=float(min_dsr),"research",f"dsr={ds}"))
    cost=report.get("cost_stress_tuned",{}) or {}; stress=cost.get("2.0") or cost.get("2"); s_ok=None if not isinstance(stress,dict) else float(stress.get("total_return",stress.get("net_compounded_return",0.0)) or 0.0)>0.0; gates.append(EvidenceGate("2x_cost_stress",s_ok,"research",f"2x={stress}"))
    hard=[g.name for g in gates if g.severity=="hard" and g.passed is False]; fail=[g.name for g in gates if g.severity=="research" and g.passed is False]; unknown=[g.name for g in gates if g.severity=="research" and g.passed is None]; score=sum(g.passed is True for g in gates)/max(1,len(gates))
    verdict="FAIL" if hard else ("INSUFFICIENT_EVIDENCE" if fail or score<0.70 else ("PROMISING_BUT_UNPROVEN" if unknown or score<0.90 else "STRONG_RESEARCH_EVIDENCE"))
    return {"verdict":verdict,"evidence_score":float(score),"hard_failed":hard,"research_failed":fail,"research_unknown":unknown,"gates":[g.to_dict() for g in gates],"promotion_allowed":False}
