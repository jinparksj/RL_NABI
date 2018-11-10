from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib
import json
import pickle
import base64
import errno
import inspect

from .tabulate import tabulate




_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='w')


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode

def get_snapshot_gap():
    return _snapshot_gap

def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap

def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def log(s, with_prefix=True, with_timestamp=True, color=None):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    if color is not None:
        out = colorize(out, color)
    if not _log_tabular_only:
        # Also logger to stdout
        print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split('\n'):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd, fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


# def log_parameters(log_file, args, classes):
#     log_params = {}
#     for param_name, param_value in args.__dict__.items():
#         if any([param_name.startswith(x) for x in list(classes.keys())]):
#             continue
#         log_params[param_name] = param_value
#     for name, cls in classes.items():
#         if isinstance(cls, type):
#             params = get_all_parameters(cls, args)
#             params["_name"] = getattr(args, name)
#             log_params[name] = params
#         else:
#             log_params[name] = getattr(cls, "__kwargs", dict())
#             log_params[name]["_name"] = cls.__module__ + "." + cls.__class__.__name__
#     mkdir_p(os.path.dirname(log_file))
#     with open(log_file, "w") as f:
#         json.dump(log_params, f, indent=2, sort_keys=True)


def stub_to_json(stub_sth):
    from schema.launch_exp import instrument
    if isinstance(stub_sth, instrument.StubObject):
        assert len(stub_sth.args) == 0
        data = dict()
        for k, v in stub_sth.kwargs.items():
            data[k] = stub_to_json(v)
        data["_name"] = stub_sth.proxy_class.__module__ + "." + stub_sth.proxy_class.__name__
        return data
    elif isinstance(stub_sth, instrument.StubAttr):
        return dict(
            obj=stub_to_json(stub_sth.obj),
            attr=stub_to_json(stub_sth.attr_name)
        )
    elif isinstance(stub_sth, instrument.StubMethodCall):
        return dict(
            obj=stub_to_json(stub_sth.obj),
            method_name=stub_to_json(stub_sth.method_name),
            args=stub_to_json(stub_sth.args),
            kwargs=stub_to_json(stub_sth.kwargs),
        )
    elif isinstance(stub_sth, instrument.BinaryOp):
        return "binary_op"
    elif isinstance(stub_sth, instrument.StubClass):
        return stub_sth.proxy_class.__module__ + "." + stub_sth.proxy_class.__name__
    elif isinstance(stub_sth, dict):
        return {stub_to_json(k): stub_to_json(v) for k, v in stub_sth.items()}
    elif isinstance(stub_sth, (list, tuple)):
        return list(map(stub_to_json, stub_sth))
    elif type(stub_sth) == type(lambda: None):
        if stub_sth.__module__ is not None:
            return stub_sth.__module__ + "." + stub_sth.__name__
        return stub_sth.__name__
    elif "theano" in str(type(stub_sth)):
        return repr(stub_sth)
    return stub_sth


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {'$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name}
        return json.JSONEncoder.default(self, o)


def log_parameters_lite(log_file, args):
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        log_params[param_name] = param_value
    if args.args_data is not None:
        stub_method = pickle.loads(base64.b64decode(args.args_data))
        method_args = stub_method.kwargs
        log_params["json_args"] = dict()
        for k, v in list(method_args.items()):
            log_params["json_args"][k] = stub_to_json(v)
        kwargs = stub_method.obj.kwargs
        for k in ["baseline", "env", "policy"]:
            if k in kwargs:
                log_params["json_args"][k] = stub_to_json(kwargs.pop(k))
        log_params["json_args"]["algo"] = stub_to_json(stub_method.obj)
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True, cls=MyEncoder)


def log_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    if hasattr(variant_data, "dump"):
        variant_data = variant_data.dump()
    variant_json = stub_to_json(variant_data)
    with open(log_file, "w") as f:
        json.dump(variant_json, f, indent=2, sort_keys=True, cls=MyEncoder)


def record_tabular_misc_stat(key, values, placement='back'):
    if placement == 'front':
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
    if len(values) > 0:
        record_tabular(prefix + "Average" + suffix, np.average(values))
        record_tabular(prefix + "Std" + suffix, np.std(values))
        record_tabular(prefix + "Median" + suffix, np.median(values))
        record_tabular(prefix + "Min" + suffix, np.min(values))
        record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        record_tabular(prefix + "Average" + suffix, np.nan)
        record_tabular(prefix + "Std" + suffix, np.nan)
        record_tabular(prefix + "Median" + suffix, np.nan)
        record_tabular(prefix + "Min" + suffix, np.nan)
        record_tabular(prefix + "Max" + suffix, np.nan)


#FROM CONSOLE.PY

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

#FROM autoargs.py

#???????????????????????????????
# def get_all_parameters(cls, parsed_args):
#     prefix = _get_prefix(cls)
#     if prefix is None or len(prefix) == 0:
#         raise ValueError('Cannot retrieve parameters without prefix')
#     info = _get_info(cls)
#     if inspect.ismethod(cls.__init__):
#         spec = inspect.getargspec(cls.__init__)
#         if spec.defaults is None:
#             arg_defaults = {}
#         else:
#             arg_defaults = dict(list(zip(spec.args[::-1], spec.defaults[::-1])))
#     else:
#         arg_defaults = {}
#     all_params = {}
#     for arg_name, arg_info in info.items():
#         prefixed_name = prefix + arg_name
#         arg_value = None
#         if hasattr(parsed_args, prefixed_name):
#             arg_value = getattr(parsed_args, prefixed_name)
#         if arg_value is None and arg_name in arg_defaults:
#             arg_value = arg_defaults[arg_name]
#         if arg_value is not None:
#             all_params[arg_name] = arg_value
#     return all_params

# def _get_info(cls_or_fn):
#     if isinstance(cls_or_fn, type):
#         if hasattr(cls_or_fn.__init__, '_autoargs_info'):
#             return cls_or_fn.__init__._autoargs_info
#         return {}
#     else:
#         if hasattr(cls_or_fn, '_autoargs_info'):
#             return cls_or_fn._autoargs_info
#         return {}