from .cognition import CognitionEngine
class DeepEvolution(CognitionEngine):
    def run(self,df,hypotheses=None): return self.run_evolution(df,hypotheses)
