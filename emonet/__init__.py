import pathlib

HOME = pathlib.Path("~").expanduser()
ROOT = pathlib.Path(__file__).parents[1]
HERE = pathlib.Path(__file__).parents[0]
DATA_DIR = HOME.joinpath("emonet-data")
PRETRAINED = ROOT.joinpath(".pretrained")

SAMPLE_RATE = 16000
EMOTIONS = ["anger", "fear", "sadness"]  # removed happy and neutral per Desmond
RATINGS = ["none", "low", "medium", "high"]
THERAPISTS = ["Michelle Lyn", "Pegah Moghaddam", "Sedara Burson", "Yared Alemu"]

# NOTE these weights are AVERAGE across ALL therapists
ANGER_WGTS = [1.0, 1.38, 2.05, 8.75]
FEAR_WGTS = [1.46, 1.31, 1.0, 2.4]
SAD_WGTS = [2.13, 1.41, 1.0, 1.36]

AGG_WEIGHTS = {"anger": ANGER_WGTS, "fear": FEAR_WGTS, "sadness": SAD_WGTS}

# NOTE these are PER THERAPIST; NaNs filled with `100.0`
WEIGHTS = {
    "Michelle Lyn": {
        "anger": [1.0, 1.0809248554913296, 1.7314814814814814, 4.155555555555556],
        "fear": [1.0, 1.0108303249097472, 1.0, 1.4814814814814814],
        "sadness": [1.3304721030042919, 1.0, 1.0801393728222994, 1.5816326530612246],
    },
    "Pegah Moghaddam": {
        "anger": [1.0, 1.4142857142857144, 1.6875, 5.711538461538462],
        "fear": [1.425531914893617, 1.2761904761904763, 1.0, 3.8840579710144927],
        "sadness": [5.19672131147541, 2.9626168224299065, 1.0, 1.268],
    },
    "Sedara Burson": {
        "anger": [1.0, 23.525000000000002, 117.625, 100.0],
        "fear": [1.0, 1.4271523178807948, 1.8903508771929827, 15.392857142857144],
        "sadness": [1.0, 1.1828793774319069, 1.316017316017316, 1.5431472081218276],
    },
    "Yared Alemu": {
        "anger": [39.0, 1.0, 1.4885496183206106, 13.295454545454545],
        "fear": [78.0, 2.493150684931507, 1.0, 2.060377358490566],
        "sadness": [35.69230769230769, 1.8861788617886182, 1.0, 1.4777070063694269],
    },
}
