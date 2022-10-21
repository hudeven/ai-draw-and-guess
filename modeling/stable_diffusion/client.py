import logging
import os
import sys
from argparse import Namespace
from typing import Any, Dict

from stability_sdk.client import *

print("test")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Set up logging for output to console.
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

STABILITY_HOST = os.getenv("STABILITY_HOST", "grpc.stability.ai:443")
STABILITY_KEY = os.getenv("STABILITY_KEY", "")

if not STABILITY_HOST:
    logger.warning("STABILITY_HOST environment variable needs to be set.")
    sys.exit(1)

if not STABILITY_KEY:
    logger.warning(
        "STABILITY_KEY environment variable needs to be set. You may"
        " need to login to the Stability website to obtain the"
        " API key."
    )
    sys.exit(1)

    
def build_request_dict(cli_args: Namespace) -> Dict[str, Any]:
    """
    Build a Request arguments dictionary from the CLI arguments.
    """
    return {
        "height": cli_args.height,
        "width": cli_args.width,
        "start_schedule": cli_args.start_schedule,
        "end_schedule": cli_args.end_schedule,
        "cfg_scale": cli_args.cfg_scale,
        "sampler": get_sampler_from_str(cli_args.sampler),
        "steps": cli_args.steps,
        "seed": cli_args.seed,
        "samples": cli_args.num_samples,
        "init_image": cli_args.init_image,
        "mask_image": cli_args.mask_image,
    }


def post_process(images):
    res = []
    for path, artifact in images:
        if artifact.type == generation.ARTIFACT_IMAGE:
            res.append(artifact.binary)
    return res


def predict(
        prompt: str,
        height:int = 512, 
        width: int = 512,
        start_schedule: float = 0.5,
        end_schedule: float = 0.01,
        cfg_scale: float = 7.0,
        sampler: str = "k_lms",
        steps: int = 50,
        seed: int = 0,
        prefix: str = "generation_",
        no_store: bool = False,
        num_samples: int = 1,
        show: bool = False,
        engine: str = "stable-diffusion-v1-5",
        init_image: str = None,
        mask_image: str = None,
    ):
    
    args = Namespace(
        prompt=prompt,
        height=height,
        width=width,
        start_schedule=start_schedule,
        end_schedule=end_schedule,
        cfg_scale=cfg_scale,
        sampler=sampler,
        steps=steps,
        seed=seed,
        prefix=prefix,
        no_store=no_store,
        num_samples=num_samples,
        show=show,
        engine=engine,
        init_image=init_image,
        mask_image=mask_image,
    )
    print(f"prompt: {args.prompt}")

    request = build_request_dict(args)

    stability_api = StabilityInference(
        STABILITY_HOST, STABILITY_KEY, engine=args.engine, verbose=True
    )

    answers = stability_api.generate(args.prompt, **request)
    artifacts = process_artifacts_from_answers(
        args.prefix, args.prompt, answers, write=not args.no_store, verbose=True
    )
    return post_process(artifacts)



if __name__ == "__main__":
    predict("a cat in water")