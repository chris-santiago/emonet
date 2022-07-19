"""
Module to run data setup scripts.

To run via CLI:

`python data_setup.py <max_workers>`

Examples
--------
`python data_setup.py 8`

"""
import sys

import emonet.batch_resample
import emonet.batch_vad
import emonet.data_prep
import emonet.dir_setup
import emonet.m4a_to_wav
import emonet.wav_splitter

MAX_WORKERS = 8  # Change based on system/container

if __name__ == "__main__":
    try:
        max_workers = int(sys.argv[1])  # if running via CLI
    except IndexError:
        max_workers = MAX_WORKERS  # if running via IDE

    emonet.dir_setup.main()
    emonet.m4a_to_wav.main(max_workers=max_workers)
    emonet.batch_resample.main(max_workers=max_workers)
    emonet.batch_vad.main()
    emonet.data_prep.main()
    emonet.wav_splitter.main()
