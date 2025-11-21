from .utils import *  # noqa

try:
    from .api_call import *  # noqa
    from .lm_call import *  # noqa
    from .vlm_call import *  # noqa
except ImportError as e:
    print(f'Error: {e}')
    print(
        'You need to install necessary dependencies to use these functions from datakit.lm_call and datakit.vlm_call.'  # noqa
    )
