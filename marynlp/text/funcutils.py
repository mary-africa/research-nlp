from collections.abc import Iterable, Callable
from functools import wraps, partial
from typing import List, Any, Optional

def _skipper_exec(out, executor: Callable, skip_rule: Optional[Callable] = None):
    """Skips execution depending on the rule"""
    if skip_rule is not None:
        if skip_rule(out):
            return out
    return executor(out)


# ----------------------------------------
# for formatting the items
# ----------------------------------------


def calls(*fns: List[Callable]):    
    """calls"""
    """This is useful if the `fns` are 'pure functions'"""
    def executor(input_: Any):
        for fn in fns:
            # check if rules are rules:
            assert isinstance(fn, Callable), "Fn '%s' is not callable" % (fn.__name__)
            input_ = fn(input_)
            
        return input_        
    return executor


def flowBy(generator):
    """Build the list"""
    def _decorator(iterator_fn):
        @wraps(iterator_fn)
        def wrapper(*args, **kwargs):
            for iter_ in iterator_fn(*args, **kwargs):
                for o_ in generator(iter_):
                    yield o_
        return wrapper
    
    _decorator.__name__ = "flowBy.{}".format(_decorator.__name__)
    return _decorator

def apply(format_fn, skip_rule: Callable = None):
    """Apply"""
    def _decorator(reader_fn):
        @wraps(reader_fn)
        def wrapper(*args, **kwargs):
            out = reader_fn(*args, **kwargs)
            return _skipper_exec(out, format_fn, skip_rule=skip_rule)
        return wrapper

    _decorator.__name__ = "apply.{}".format(_decorator.__name__)
    return _decorator

def yield_forEach(fn, skip_rule: Callable = None):
    """Yield"""
    def _decorator(reader_fn):
        @wraps(reader_fn)
        def wrapper(*args, **kwargs):
            out = reader_fn(*args, **kwargs)
            assert isinstance(out, Iterable), "The output of fn '%s' is not iterable" % (reader_fn.__name__)
            for o in out:
                yield _skipper_exec(o, fn, skip_rule=skip_rule)
        return wrapper

    _decorator.__name__ = "yield_forEach.{}".format(_decorator.__name__)
    return _decorator

def forEach(fn, skip_rule: Callable = None, type_: Any = None):
    """forEach"""
    def _decorator(reader_fn):
        @wraps(reader_fn)
        def wrapper(*args, **kwargs):
            out = reader_fn(*args, **kwargs)
            assert isinstance(out, Iterable), "The output of fn '%s' is not iterable" % (reader_fn.__name__)
            iter_ = map(partial(_skipper_exec, executor=fn, skip_rule=skip_rule), out)
            if type_ is None:
                return list(iter_)
            
            return type_(iter_)
        return wrapper

    _decorator.__name__ = "forEach.{}".format(_decorator.__name__)
    return _decorator

# ------------------------------------------
# for filtering items
# ------------------------------------------

def rules(*fns: List[Callable]):    
    """This is useful if the `fns` are 'selectors'"""
    def selector(input_: Any):
        """Selectors
        Returns:
            True, if the selector is are all true. False if otherwise
        """
        select: bool = True        
        for fn in fns:
            assert isinstance(fn, Callable), "Fn '%s' is not callable" % (fn.__name__)
            if select: 
                select = select and fn(input_)
            else:
                break            
        return select        
    return selector

def filterBy(filter_rule: Callable, type_: Any = None):
    """"""
    def filter_decorator(outptter_fn):
        @wraps(outptter_fn)
        def wrapper(*args, **kwargs):
            out = outptter_fn(*args, **kwargs)
            assert isinstance(out, Iterable), "The output of fn '%s' is not iterable" % (outptter_fn.__name__)
            iter_  = filter(filter_rule, out)
            if type_ is None:
                return list(iter_)
            
            return type_(iter_)
        return wrapper
    return filter_decorator
