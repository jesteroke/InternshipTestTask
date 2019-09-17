"""Human segmentation project main function"""
import argparse
import json

from lib.trainer import Trainer  # type: ignore


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        required=True
    )

    arguments = parser.parse_args()

    return arguments


def main(args):
    """
    Read config file and run the experiment.

    Parameters
    ----------
    config : str
        path to config JSON file.

    Returns
    -------
    None

    Notes
    -----
    See configs/example.json
    """
    with open(args.config, 'r') as f:
        config = json.load(f)

    mode = config['mode']

    trainer = Trainer(config)

    if mode == 'train':
        trainer.train()

    elif mode == 'test':
        trainer.test()

    return


if __name__ == '__main__':
    main(parse_args())
