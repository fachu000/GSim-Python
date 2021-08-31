""" gsim_conf.py is intended to be ignored in .gitignore. It is a configuration
file specific to each user.

gsim_conf-base.py can be used to create a template for gsim_conf.py specific to
each project. """

# Select an experiment file:
module_name = "experiments.example_experiments"

# GFigure
import gsim.gfigure

gsim.gfigure.title_to_caption = True
gsim.gfigure.default_figsize = (5.5, 3.5) # `None` to let plt choose.

#log.setLevel(logging.DEBUG)
import logging.config

cfg = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '{levelname}:{name}:{module}: {message}',
            'style': '{',
        },
        'standard': {
            'format': '{levelname}:{asctime}:{name}:{module}: {message}',
            'style': '{',
        },
        'verbose': {
            'format':
            '{levelname}:{asctime}:{name}:{module}:{process:d}:{thread:d}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        # 'file': {
        #     'level': 'INFO',
        #     'class': 'logging.FileHandler',
        #     'filename': os.path.join(BASE_DIR, LOGGING_DIR, 'all.log'),
        #     'formatter': 'standard'
        # },
        'console': {  # This one is overridden in settings_server.py
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'experiments': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': True,
        },
    }
}
logging.config.dictConfig(cfg)