import argparse
import sys
import loguru

from wargame_rl.wargame.model.dqn.agent import Agent

if __name__ == "__main__":
    logger = loguru.logger
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train", type=bool, default=True, help="Run training if True, otherwise run inference.")
    arg_parser.add_argument("--render", type=bool, default=False, help="Render the environment if True.")
    args = arg_parser.parse_args()

    agent = Agent(
        hyperparameter_set="wargame",
    )
    logger.info(f"Starting {'training' if args.train else 'inference'} with rendering set to {args.render}.")
    agent.run(
        is_training=args.train,
        render=args.render
    )