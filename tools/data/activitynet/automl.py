# -*- coding: utf-8 -*-
import os
import re
import logging
import sys
import requests
import functools
import json
import time
import random
import traceback
from datetime import datetime
from copy import deepcopy
from typing import Any, Union

# 日志相关
logger = logging.getLogger('AutoML_API')
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

PBT_LOG = """
*****************************************************************************
Set PBT SAVE/LOAD PATH {}:
    {}
*****************************************************************************
"""


class StatusCode:
    """状态码"""

    Success = 0
    Fail = 1
    Wait = 2


def retry(times=2, delay=0.1):
    """函数重试装饰器"""

    def wrapper(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            for i in range(times):
                try:
                    return function(*args, **kwargs)
                except Exception:
                    logger.info("[retry/%d] execute func %s with exception %s",
                                i, function.__name__, traceback.format_exc())
                    time.sleep(delay)
            return function(*args, **kwargs)

        return wrapped

    return wrapper


@retry(times=5, delay=30)
def http_get(url: str, params: dict = None, timeout=None):
    resp = requests.get(url, params=params, timeout=timeout)
    if not resp or not resp.ok:
        raise ValueError("invalid http get resp {}".format(resp))
    return resp


class AutoMLAPI:
    """automl api接口管理"""

    @staticmethod
    def get_host(env: str, name: str) -> str:
        """根据服务名称获取host，如果有多个，则随机返回一个"""
        headers = {'Content-Type': 'application/json'}
        data = {"type": 1, "service": {"namespace": env, "name": name}}
        resp = requests.post("http://polaris-discover.oa.com:8080/v1/Discover",
                             headers=headers, data=json.dumps(data))
        if not resp or not resp.ok:
            logger.info("AutoMLAPI get host failed. env {}, name {}, resp {}".format(env, name, resp))
            raise Exception("AutoMLAPI get host failed.")
        data = resp.json()
        if "instances" not in data:
            logger.info("AutoMLAPI get host failed. env {}, name {}, resp {}".format(env, name, resp))
            raise Exception("AutoMLAPI get host failed.")
        # 随机返回一个host
        instances = list(filter(lambda item: item["healthy"] and not item["isolate"], data["instances"]))
        if len(instances) == 0:
            # 无可用host，返回None
            logger.info("AutoMLAPI get host failed. not host valid. env {}, name {}, instances {}"
                        .format(env, name, data["instances"]))
            raise Exception("AutoMLAPI get host failed.")
        item = random.choice(instances)
        res = item["host"] + ":" + str(item["port"])
        logger.info("AutoMLAPI get host success. env {}, name {}, host {}".format(env, name, res))
        return res

    @staticmethod
    def get_domain(env: str, name: str) -> str:
        """根据环境参数返回域名"""
        res = "venus.oa.com/automlapi" if env == "Production" else AutoMLAPI.get_host(env, name)
        logger.info("AutoMLAPI get host success. env {}, domain {}".format(env, res))
        return res

    @staticmethod
    def query_experiment_params(host: str, uuid: str) -> dict:
        """根据uuid获取实验信息"""
        url = f"http://{host}/experiment"
        resp = http_get(url, params=dict(uuid=uuid), timeout=5)
        logger.info("AutoMLAPI query experiment info. uuid {}, resp {}".format(uuid, resp))
        if not resp or not resp.ok:
            raise Exception('AutoMLAPI query experiment info with invalid resp.')
        return resp.json()

    @staticmethod
    def report_metrics(host: str, uuid: str, trial_id: int, name: str, value: Union[float, int], seq: int,
                       report_type: int) -> bool:
        """上报metrics给automl，用于评估超参训练效果，进行下一轮调参的优化"""
        url = f"http://{host}/automl/reportMetricData"
        params = dict(uuid=uuid, trialID=trial_id, value=value, name=name,
                      jobID=trial_id, sequence=seq, type=report_type)
        resp = http_get(url, params=params, timeout=5)
        logger.info("AutoMLAPI report metrics. params {}, resp {}".format(params, resp))
        return resp and resp.ok

    @staticmethod
    def query_trial_params(host: str, uuid: str, trial_id: int) -> (StatusCode, str):
        """查询trial参数，避免环境变量过长"""
        url = f"http://{host}/trial-jobs"
        params = {"uuid": uuid, 'trialID': int(trial_id)}
        try:
            resp = http_get(url, params=params, timeout=5)
            logger.info("AutoMLAPI query trial params. req params {}, resp {}".format(params, resp))
            if not resp or not resp.ok:
                raise Exception('AutoMLAPI query trial params with invalid resp.')
            content = resp.json()['data'][0]['hyperParameters'][0]
            logger.info("AutoMLAPI query trial params. req params {}, resp params {}".format(params, content))
            return StatusCode.Success, content
        except Exception as e:
            logger.info("AutoMLAPI query trial params with exception", traceback.format_exc())
            return StatusCode.Wait, str(e)

    @staticmethod
    def get_next_trial_params(host: str, uuid: str, last_trial_id: int) -> (StatusCode, str):
        """获取下一组trial参数"""
        url = f"http://{host}/automl/getNextParams"
        params = {"uuid": uuid, "lastTrialID": last_trial_id}
        try:
            resp = http_get(url, params=params, timeout=300)
            logger.info("AutoMLAPI get next trial params. req params {}, resp {}".format(params, resp))
            if not resp or not resp.ok:
                raise Exception('AutoMLAPI get next params with invalid resp.')
            ret = resp.json()["ret"]
            if ret.get("code") == 0:
                return StatusCode.Success, resp.json()["content"]
            elif ret.get("msg") == "waiting population next round":
                return StatusCode.Wait, ret.get("msg")
            else:
                return StatusCode.Fail, ret.get("msg")
        except Exception as e:
            logger.info("AutoMLAPI get next trial params with exception", traceback.format_exc())
            return StatusCode.Wait, str(e)

    @staticmethod
    def report_trial_fail(host: str, uuid: str, trial_id: int, fail_type: int, fail_msg: str) -> bool:
        """上报某个trial失败"""
        url = f"http://{host}/automl/report-trial-fail"
        params = {"uuid": uuid, 'trialID': trial_id, "failType": fail_type, "failMsg": fail_msg}
        resp = http_get(url, params=params, timeout=5)
        logger.info("AutoMLAPI report trial fail. req params {}, resp {}".format(params, resp))
        return resp and resp.ok

    @staticmethod
    def set_run_mode(host: str, uuid: str, run_mode: int) -> bool:
        """设置运行模式"""
        url = f"http://{host}/automl/set-run-mode"
        params = {"uuid": uuid, 'runMode': run_mode}
        resp = http_get(url, params=params, timeout=5)
        logger.info("AutoMLAPI set run mode. req params {}, resp {}".format(params, resp))
        return resp and resp.ok


class ParamsManager(object):
    def __init__(self):
        self._params = dict()

    def load_from_path(self, path):
        """从文件中加载变量"""
        if not os.path.exists(path):
            logger.info("ParamsManager load from path failed. path {} not exists.".format(path))
            return False
        with open(path, "r") as f:
            self.add(json.load(f))
            logger.info("ParamsManager load from path success. path {}".format(path))
            return True

    def get_params(self) -> dict:
        return self._params

    def add(self, params):
        """添加变量对"""
        self._params.update(params)

    def get(self, key: str, default_value: Any = None) -> Any:
        """获取变量值"""
        value = self._params.get(key)  # 从配置里获取
        if value is None:
            value = os.environ.get(key)  # 从环境变量获取
        if value is None:
            value = default_value
        return value


class Experiment:
    def __init__(self, attr_dic):
        self._tuner = attr_dic['params']['tune']['builtinTunerName']
        self._name = attr_dic['params']['experimentName']
        self._start_time = attr_dic['startTime']
        self._pbt_path = self._gen_pbt_path()
        if self._pbt_path:
            logger.info(PBT_LOG.format('Succeed', self._pbt_path))

    def get_pbt_path(self) -> str:
        return self._pbt_path

    def set_pbt_path(self, path: str):
        self._pbt_path = os.path.abspath(path)

    def _gen_pbt_path(self) -> str:
        # 如果是pbt/PBA Tuner,生成默认路径
        if self._tuner not in ['PBTTuner', "PBATuner"]:
            return ""
        exp_flag = re.sub('[\s/]', "", self._name)
        time_flag = datetime.fromtimestamp(int(self._start_time) / 1000).strftime('%Y%m%d%H%M')
        api_path, _ = os.path.split(os.path.abspath(sys.argv[0]))
        return os.path.abspath(os.path.join(f"{api_path}/checkpoint_dir/{exp_flag}_{time_flag}"))

    def __repr__(self):
        return f"tuner:{self._tuner} exp_name:{self._name}, start_time:{self._start_time}, pbt_path:{self._pbt_path}"


class ReportManager:
    def __init__(self):
        self._global_params = None
        self._experiment = None
        self._host = None
        self._uuid = None
        # trial相关参数
        self._trial_id = None
        self._trial_params = None
        self._sequence = 0
        self._is_trial_finish = False  # 上报了final metric则视为trial运行结束
        self._last_report_metric = None  # 最近一次上报的metric数据
        self._is_first_trial = True
        self._history_trial_ids = list()

    def init_global_params(self):
        self._global_params = ParamsManager()

    def init_experiment(self):
        polaris_env = self._global_params.get("polaris_env")
        polaris_name = self._global_params.get("polaris_name")
        self._host = AutoMLAPI.get_domain(polaris_env, polaris_name)
        self._uuid = self._global_params.get("UUID")
        self._experiment = Experiment(AutoMLAPI.query_experiment_params(self._host, self._uuid))

    def _init_trial(self, trial_id, params):
        self._trial_id = trial_id
        self._trial_params = ParamsManager()
        self._trial_params.add(params)
        self._trial_params.add({"trial_id": trial_id})
        self._sequence = 0
        self._is_trial_finish = False
        self._last_report_metric = None
        self._is_first_trial = False

    def set_exp_pbt_path(self, path: str):
        self._experiment.set_pbt_path(path)

    def query_trial_params(self) -> (StatusCode, str):
        # 从环境变量拿到的trial_id，首次起trial时使用
        global_trial_id = self._global_params.get("TrialInd")
        if self._is_first_trial and global_trial_id is not None:
            return AutoMLAPI.query_trial_params(self._host, self._uuid, global_trial_id)
        last_trial_id = -1 if not self._history_trial_ids else self._history_trial_ids[-1]
        return AutoMLAPI.get_next_trial_params(self._host, self._uuid, last_trial_id)

    def update_trial_params(self) -> bool:
        while True:
            logger.info("[init_trial_params] start get next trial")
            status, data = self.query_trial_params()
            if status == StatusCode.Success:
                json_data = json.loads(data)
                self._init_trial(json_data["parameter_id"], json_data["parameters"])
                logger.info("[init_trial_params] with trial id %d" % self._trial_id)
                logger.info('===============')
                logger.info(f'This hyperParameters : {self._trial_params.get_params()}')
                logger.info('===============')
                return True
            elif status == StatusCode.Fail:
                logger.info("[init_trial_params] failed %s" % data)
                return False
            elif status == StatusCode.Wait:
                logger.info("[init_trial_params] wait sometime %s" % data)
                time.sleep(5)
            else:
                logger.info("[init_trial_params] unknown status {}, msg {}".format(status, data))
                return False

    def get_trial_params(self) -> dict:
        return self._trial_params.get_params()

    def gen_sequence_id(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq

    def get_save_checkpoint_dir(self):
        save_dir = self._trial_params.get('save_checkpoint_dir')
        return os.path.join(self._experiment.get_pbt_path(), save_dir) if save_dir else None

    def get_load_checkpoint_dir(self):
        load_dir = self._trial_params.get('load_checkpoint_dir')
        return os.path.join(self._experiment.get_pbt_path(), load_dir) if load_dir else None

    def report_metrics(self, name: str, value: Union[float, int], report_type: int, remote: bool = True) -> bool:
        # 记录最后一次上报的数据
        self._last_report_metric = (name, value)
        # 不需要上报到远端，直接返回成功
        if not remote:
            return True
        # 上报到automl_server
        seq_id = 0 if report_type == 1 else self.gen_sequence_id()
        succ = AutoMLAPI.report_metrics(self._host, self._uuid, self._trial_id, name, value, seq_id, report_type)
        # 上报final成功视为trial正常结束
        self._is_trial_finish = self._is_trial_finish or (report_type == 1 and succ)
        return succ

    def report_trial_fail(self, fail_type: int = 0, fail_msg: str = "") -> bool:
        return AutoMLAPI.report_trial_fail(self._host, self._uuid, self._trial_id, fail_type, fail_msg)

    def update_params(self, params: Union[dict, object]):
        """更新训练参数"""

        def add_param(k, v):
            if isinstance(params, dict):
                params[k] = v
            else:
                params.__setattr__(k, v)

        # trial参数
        for key, value in self.get_trial_params().items():
            add_param(key, value)
        # checkpoint
        save_dir = self.get_save_checkpoint_dir()
        if save_dir:
            add_param("save_checkpoint_dir", save_dir)
        load_dir = self.get_load_checkpoint_dir()
        if load_dir:
            add_param("load_checkpoint_dir", load_dir)
        logger.info("[update_params] params after update {}".format(params))

    def start_trial_loop(self, exe_func, params: Union[dict, object] = None):
        AutoMLAPI.set_run_mode(self._host, self._uuid, 1)
        while self.update_trial_params():
            try:
                if params is None:
                    # 无参数时由exe_func调用update_parameter(params, False)来更新参数
                    logger.info("[start_trial_loop] start new trial")
                    exe_func()
                else:
                    # 有参数时更新参数后启动trial
                    param_dict = deepcopy(params)
                    self.update_params(param_dict)
                    logger.info("[start_trial_loop] start new trial with params {}".format(param_dict))
                    exe_func(param_dict)
            except Exception as e:
                logger.info("[start_trial_loop] with exception {}".format(e))
                logger.info(traceback.format_exc())
            finally:
                self._history_trial_ids.append(self._trial_id)
                # 如果没有上报过final metric，则将最后一次上报的metric作为final
                if not self._is_trial_finish and self._last_report_metric:
                    self.report_metrics(self._last_report_metric[0], self._last_report_metric[1], 1)
                # 如果trial没有正常结束，上报fail状态
                if not self._is_trial_finish:
                    self.report_trial_fail()
        logger.info("[start_trial_loop] loop exit...")


# 全局句柄
report_manager = ReportManager()
report_manager.init_global_params()
report_manager.init_experiment()

"""
以下函数供用户调用
"""


def start_trial_loop(exe_func, params: Union[dict, object] = None):
    """开启训练循环
    :param exe_func: [func] 训练函数，调用方式为exe_func(params)
    :param params: [dict, object] 训练所需参数
    """
    report_manager.start_trial_loop(exe_func, params)


def update_parameter(params: Union[dict, object], update_trial_params: bool = True):
    """更新训练参数"""
    if update_trial_params:
        report_manager.update_trial_params()
    report_manager.update_params(params)


def set_pbt_path(pbt_path: str):
    """指定PBT临时文件路径"""
    report_manager.set_exp_pbt_path(pbt_path)


def report_final_result(name: str, value: Union[float, int]) -> bool:
    """上报训练完成后metric值"""
    return report_manager.report_metrics(name, value, 1)


def report_intermediate_result(name: str, value: Union[float, int], remote: bool = True) -> bool:
    """上报训练中metric值"""
    return report_manager.report_metrics(name, value, 0, remote=remote)


def report_trail_fail(fail_type: int, fail_msg: str):
    """上报trial失败原因"""
    return report_manager.report_trial_fail(fail_type, fail_msg)


def get_load_checkpoint_dir():
    """PBT使用，加载模型"""
    return report_manager.get_load_checkpoint_dir()


def get_save_checkpoint_dir():
    """PBT使用，保存模型"""
    return report_manager.get_save_checkpoint_dir()


def get_trial_params():
    """获取本次trial参数"""
    return report_manager.get_trial_params()