import argparse
import logging
import logging.config
import pickle

from dotenv import load_dotenv

load_dotenv()
LOGGER = logging.getLogger(__name__)
# load config from file
logging.config.fileConfig("logging.ini", disable_existing_loggers=False)


def main(decomp, **sbm_kwargs):
    state = decomp.run_sbm(**sbm_kwargs)
    decomp.save_sbm_res(state, **sbm_kwargs)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SBM")
    sbm_kwds = ["metric", "transfo", "rec_types", "scaler"]
    for kwd in sbm_kwds:
        parser.add_argument(f"--{kwd.replace('_', '-')}")

    parser.add_argument("--decomp-save-path", required=True)

    args = parser.parse_args()
    sbm_kwargs = vars(args)

    decomp_save_path = sbm_kwargs.pop("decomp_save_path")
    LOGGER.debug(f"loading {decomp_save_path}")
    with open(decomp_save_path, "rb") as f:
        decomp = pickle.load(f)
    LOGGER.debug(f"loaded {decomp_save_path}")

    state = main(decomp, **sbm_kwargs)
    LOGGER.debug("done")
