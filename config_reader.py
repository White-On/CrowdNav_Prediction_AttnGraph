from configparser import ConfigParser
from pathlib import Path
from argparse import Namespace

import logging

def read_config(pathConfig, listSection=[]) -> Namespace:
    config = {}
    configParser = ConfigParser()
    logging.info(f"Reading config file from {pathConfig}")
    try:
        configParser.read(pathConfig.absolute())
    except Exception as e:
        logging.error(f"Error while reading config file, probably a duplicate section")
        raise Exception(f"Error while reading config file : {e}")
    
    sections = configParser.sections() if len(listSection) == 0 else listSection
    try:
        config = {
            section: {
                key: configParser.get(section, key)
                for key in configParser.options(section)
            }
            for section in sections
        }
    except Exception as e:
        logging.error(f"Error while reading config file : {e}")
        return {}
    return Namespace(**config)