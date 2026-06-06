""" Time-series modeling toolbox """

__version__ = '1.0'

__submodules__ = {
    'optimization', 'interpolators',
    'rbf_nets', 'multimodels', 'multimodels_win'}
__all__ = list(__submodules__)

def __getattr__(attr):
    """ Return the correct module from its name, if it exists """
    # pylint: disable=import-outside-toplevel, consider-using-from-import
    if attr == 'optimization':
        import modeling.optimization as optimization
        return optimization
    if attr == 'interpolators':
        import modeling.interpolators as interpolators
        return interpolators
    if attr == 'rbf_nets':
        import modeling.rbf_nets as rbf_nets
        return rbf_nets
    if attr == 'multimodels':
        import modeling.multimodels as multimodels
        return multimodels
    if attr == 'multimodels_win':
        import modeling.multimodels_win as multimodels_win
        return multimodels_win
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

def __dir__():
    """ Add the modules of `submodules` to the list of callable variables"""
    public_symbols = globals().keys() | __submodules__
    return list(public_symbols)
