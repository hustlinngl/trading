from ai_trading_lab.config import load_settings
from ai_trading_lab.autonomous import autonomous_cycle
import argparse, json

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--query', default=None)
    args=p.parse_args()
    s=load_settings(args.config)
    print(json.dumps(autonomous_cycle(s, args.query or s.autonomous_query), indent=2, default=str))
