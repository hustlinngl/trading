from .cognition import CognitionEngine
class AutonomousGrowth(CognitionEngine):
    def cycle(self,df,query,strategy_candidates=None): return self.run_growth(df,query,strategy_candidates)
