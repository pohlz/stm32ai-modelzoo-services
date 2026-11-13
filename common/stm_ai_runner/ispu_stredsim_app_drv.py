###################################################################################
#   Copyright (c) 2026 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
ISPU simulator stredsim driver
"""
import os
from shutil import which
import subprocess
import sys

from .ai_runner import AiRunnerError, AiRunner
from .app_drv import AppDriver
from functools import reduce
import re


class IspuStredsimAppDriver(AppDriver):
    """
    Class to handle ispu application
    """

    SUPPORTED_MODES = ["cycle", "instruction"]

    def __init__(self, parent):
        self.clock = None
        self.compiler = None
        self.profiling = False
        self.profile_file = None
        self.simulation_mode = None
        self.additional_args = []

        super(IspuStredsimAppDriver, self).__init__(parent)

    def _get_device_info(self):
        """
        Create the dictionary with device info

        Returns
        -------
        dict
            A dictionary with device info
        """
        info_device = dict()
        info_device['desc'] = 'stredsim ISPU simulator'
        info_device['dev_type'] = 'SIMULATOR'
        return info_device

    def connect(self, desc=None, **kwargs):
        """
        Connect to application and configure app for profiling if specified

        Parameters
        ----------
        desc: str
            The application
        kwargs
            The other arguments
        """
        context = kwargs["context"]
        target_series = context.get_device_info().get_generic_device_name()
        if target_series == "imu22":
            self.clock = 5000
        elif target_series == "acc50":
            self.clock = 40000
            self.additional_args += ["-zol", "-fpsqrt"]
        else:
            raise AiRunnerError(f"{target_series} does not exists for ispu target")

        if context.get_option("validate.ispu_profile") is not None:
            self.profiling = True
            self.profile_file = context.get_option("validate.ispu_profile")
            self.additional_args += ["-count"]

        self.simulation_mode = context.get_option("validate.sim_mode")
        if self.simulation_mode not in IspuStredsimAppDriver.SUPPORTED_MODES:
            raise AiRunnerError(f"{self.simulation_mode} is not supported by ispu simulator")

        super().connect(desc, **kwargs)

    def _run_application(self, sample):
        """
        Run the wrapped application

        Parameters
        ----------
        sample: int
            The sample index to be evaluated (0 for inspection)

        Returns
        -------
        The output of subprocess.run and the elapsed time

        Raises
        ------
        AiRunnerError
            If stredsim is not found
        AssertionError
            If the output return code is not 0
        """
        stredsim = which('stredsim')

        if stredsim is None:
            raise AiRunnerError('stredsim not found')

        if self.simulation_mode == "cycle":
            start_arg = "-cyc_func_start"
            stop_arg = "-cyc_func_stop"
        else:
            start_arg = "-inst_func_start"
            stop_arg = "-inst_func_stop"

        lite_app_args = [ stredsim,
                          "-app", self._app,
                          "-fclk", str(self.clock // 1000),
                          start_arg, "th_signal_start",
                          stop_arg, "th_signal_stop",
                          "-argv", str(sample)
                        ] + self.additional_args
        self._logger.info('Executing %s', ' '.join(lite_app_args))
        output = subprocess.run(lite_app_args, check=False, capture_output=True, text=True)
        sys.stdout.flush()

        if output.returncode:
            stderr = output.stderr

            for line in output.stdout.splitlines():
                self._logger.error(line)

            # For memory overlap, the error message is printed on stderr
            for line in stderr.splitlines():
                self._logger.error(line)
            raise AssertionError('Error in running validation application')
        output = output.stdout.splitlines()
        elapsed_time = 0

        if sample != 0:
            cycles = int(output[-1].split(" ")[-1])
            elapsed_time = cycles / self.clock

            if self.profiling:
                n_outs = len(self.get_info()["outputs"])
                end_tag = output.index(f"__END_OUTPUT{n_outs} __")
                f_calls = output[end_tag + 4: -1]
                f_calls.sort(reverse=True, key=lambda s: int(s.split("\t")[0]))

                with open(self.profile_file, "w") as pf:
                    for line in output[end_tag + 1: end_tag + 4]:
                        pf.write(line + "\n")

                    for line in f_calls:
                        pf.write(line + "\n")

        self._logger.debug('Elapsed time is %s', str(elapsed_time))

        return output, elapsed_time

    @property
    def capabilities(self):
        """
        Return list with the capabilities

        Returns
        -------
        list
            List of capabilities
        """
        return [AiRunner.Caps.IO_ONLY]

    def short_desc(self):
        """
        Return human readable description

        Returns
        -------
        str
            Human readable description
        """
        return 'ISPU STREDSIM app' + ' (' + self._app + ')'
