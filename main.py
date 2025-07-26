import argparse
import sys

import loguru


def environment_test():
    from wargame_rl.wargame.envs.env_test import EnvTest

    # Initialize the environment with a grid size of 50 and human rendering mode
    env_test = EnvTest(size=50, render_mode="human")

    # Run 100 random actions in the environment
    env_test.run_actions(num_actions=100)


def main():
    logger = loguru.logger
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--train",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Run training if True, otherwise run inference.",
    )
    arg_parser.add_argument(
        "--render",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Render the environment if True.",
    )
    arg_parser.add_argument(
        "--env_test",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Run a test environment if True.",
    )
    args = arg_parser.parse_args()

    if args.env_test:
        logger.info("Running environment test...")
        environment_test()
        return


if __name__ == "__main__":
    main()
