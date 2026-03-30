import sys  # nopep8
import os  # nopep8

from dataclasses import dataclass  # nopep8


@dataclass
class EnvSettings():
    # Causes the artificial data loader to only load the first few sequences
    # for quick debugging.
    loadfast = False

    wandb_enabled = False

    gui_enabled = True


env_settings = EnvSettings()


def try_get_env_setting(name, setting):
    thing = os.getenv(name)
    if thing is None:
        print(
            "Environment variable {} not set. Using default value {}".format(
                name,
                setting))
        return setting
    else:
        print("Environment variable {} set to {}".format(name, thing))
        return thing


env_settings.loadfast = bool(
    int(try_get_env_setting("AD4_LOADFAST", env_settings.loadfast)))
env_settings.wandb_enabled = bool(
    int(try_get_env_setting("AD4_ENABLEWANDB", env_settings.wandb_enabled)))
env_settings.gui_enabled = bool(
    int(try_get_env_setting("AD4_ENABLEGUI", env_settings.gui_enabled)))
