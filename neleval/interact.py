def run_ipython(local):
    try:
        from IPython.frontend.terminal.embed import TerminalInteractiveShell
        shell = TerminalInteractiveShell(user_ns=local)
        shell.mainloop()
    except ImportError:
        # IPython < 0.11
        # Explicitly pass an empty list as arguments, because otherwise
        # IPython would use sys.argv from this script.
        from IPython.Shell import IPShell
        shell = IPShell(argv=[], user_ns=local)
        shell.mainloop()


def run_bpython(local):
    import bpython
    bpython.embed(locals_=local)


def run_python(local):
    import code
    try:
        import readline
    except ImportError:
        pass
    else:
        import rlcompleter
        readline.set_completer(rlcompleter.Completer(local).complete)
        readline.parse_and_bind('tab:complete')
    code.interact(local=local)


lookup = {'ipython': run_ipython, 'python': run_python, 'bpython': run_bpython}


def embed_shell(local, shell=None):
    if shell is None:
        options = [run_ipython, run_bpython, run_python]
    else:
        options = [lookup[shell]]
    e = None
    for fn in options:
        try:
            fn(local)
            return
        except ImportError as e:
            pass
    raise e
