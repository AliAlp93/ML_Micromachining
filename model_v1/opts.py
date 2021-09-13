"""
Program options and settings, settable via command line args using argparse.

Use:
In command line to set an option:
```
python file_name.py --some-arg some_value
```

In a file which requires opts.
```
from opts import get_opts
opts = get_opts()
# ...
assert opts.some_arg == 'some_value'
```
"""

import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='24789 — ML for Machining')

    # Feature detection (requires tuning)
    parser.add_argument('--data-dir', type=str, default='../data/ForLSTM',
                        help='Where data files should be loaded from.')

    # Training:
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Default number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Default learning rate for training.')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Size of each training batch.')
    parser.add_argument('--validation-split', type=float, default=0.3,
                        help='Proportion of total dataset which should be used for validation.')
    parser.add_argument('--test-split', type=float, default=0.0,
                        help='Proportion of total dataset which should be used for final testing.')

    parser.add_argument('--channel-forecast-start', type=int, default=7,
                        help='Which channel number to start forecasting at.')
    parser.add_argument('--channel-forecast-step', type=int, default=7,
                        help='How many channels to step by when increasing forecast distance.')

    opts = parser.parse_args()

    return opts
