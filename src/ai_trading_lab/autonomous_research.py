from .cognition import CognitionEngine,Hypothesis,CandidateReport
class AutonomousResearchLoop(CognitionEngine):
    def run_cycle(self,df,query): return self.run_research(df,query)
__all__=['AutonomousResearchLoop','Hypothesis','CandidateReport']
