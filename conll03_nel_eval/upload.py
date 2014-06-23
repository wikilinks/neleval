"""Tool to deposit a system output as a github pull request"""
from __future__ import print_function

import textwrap
import tempfile
import os
from collections import namedtuple
from subprocess import check_output, CalledProcessError
import shutil
import getpass
import __builtin__

from .formats import Unstitch

if hasattr(__builtin__, 'raw_input'):
    # Py2k compatibility
    input = raw_input

try:
    import github3
except ImportError:
    github3 = None
else:
    from github3 import login

from .constants import GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, REFERENCE_OUTPUT_FMT, REFERENCE_README_FMT
from .utils import log


MENTIONS_NAME = {'gold': 'goldmentions',
                 'linked': 'goldlinkedmentions',
                 'system': 'systemmentions'}


Field = namedtuple('Field', 'var prompt desc validate invalid_msg')

PROMPT_FIELDS = [
    Field('system_name', 'a unique name for your system',
          'This will be used to name the files pertaining to your system in the repository.',
          lambda x: x.strip(), ''),  # TODO: check for likely uniqueness
    Field('output_path', 'the path of your system output on testb (stitched)', '',
          os.path.isfile, 'File not found'),
    Field('readme_path', 'the path of a description of your system', '',
          os.path.isfile, 'File not found'),
    Field('mentions', 'use of gold or system mentions (gold/linked/system)',
          textwrap.dedent('''
          gold: your system receives gold linked and unlinked mentions
          linked: your system only received gold linked mentions
          system: your system generates its own mentions to link''').strip(),
          lambda x: x in ('gold', 'linked', 'system'),
          'Please enter one of gold/linked/system'),
    Field('mapping_id', 'the target ID mapping used (e.g. api20140227, unmapped)',
          '', lambda x: x, ''),  # TODO validate fully
    Field('gh_username', 'your github username',
          '', lambda x: x, ''),
    Field('gh_password', 'your github password',
          '', lambda x: x, ''),
]

FIELD_MAP = {pf.var: pf for pf in PROMPT_FIELDS}


def two_factor_cb(prompt):
    return input(prompt)


class Upload(object):
    """Upload a system output to a centralised repository

    Creates a pull request on GitHub. For command-line use, only the system
    name must be given on the command-line and other fields will be prompted.

    Git must be on PATH and the github3 Python module must be installed.
    """

    def __init__(self, system_name=None, output_path=None, readme_path=None,
                 mapping_id=None, mentions=None, gh_username=None, gh_password=None):
        if github3 is None:
            raise ImportError('Could not import package github3. Please install from PyPI using `pip install github3`')
        try:
            check_output(['git', 'help'])
        except (OSError, CalledProcessError):
            raise OSError('Could not execute a git command. You must have git installed and its commands on PATH.')

        for k, v in self.prompt(locals()).items():
            exec '{} = {!r}'.format(k, v)

        self.gh_username = gh_username
        self.gh = login(gh_username, gh_password, two_factor_callback=two_factor_cb)
        self.system_name = system_name
        self.output_path = output_path
        assert os.path.isfile(output_path)  # TODO: format validation
        self.readme_path = readme_path
        assert os.path.isfile(readme_path)
        self.mapping_id = mapping_id
        self.mentions = mentions

    @classmethod
    def prompt(cls, data):
        required = [pf for pf in PROMPT_FIELDS if data.get(pf.var) is None]
        if not required:
            return
        out = {}
        for pf in required:
            resp = ''
            while not resp.strip():
                prompt = input if 'password' not in pf.var else getpass.getpass
                resp = prompt('\n' + pf.prompt + (' (? for help)' if pf.desc else '') + ': ')
                if resp.strip() == '?':
                    print(pf.desc or '[sorry, no further help is available for this prompt]')
                    resp = ''
                else:
                    if not pf.validate(resp):
                        print(pf.invalid_msg)
                        resp = ''
            out[pf.var] = resp.strip()
        return out
        
    def __call__(self):
        unstitch = Unstitch(self.output_path)
        try:
            unstitched_output = unstitch()
        except Exception as e:
            raise ValueError('The data is not in the correct format; '
                             'it may not be stitched for evaluation. '
                             '(Error was: {!r})'.format(e))


        repo = self.gh.repository(GITHUB_OWNER, GITHUB_REPO)
        log.info('Creating a personal fork of https://github.com/{}/{}'.format(GITHUB_OWNER, GITHUB_REPO))
        fork = repo.create_fork()
        fork_url = fork.clone_url
        if '@' not in fork_url:
            fork_url = fork_url.replace('://', '://' + self.gh_username + '@', 1)
        root = tempfile.mkdtemp(suffix='.tmp', prefix='{}-upload-{}')
        # TODO check clone_url includes username, and perhaps consider another form
        log.info('Checking out fork from {} into {}'.format(fork_url, root))
        check_output(['git', 'clone', '--branch', GITHUB_BRANCH, fork_url, root])
        git_args = ['git', '--git-dir', os.path.join(root, '.git'), '--work-tree', root]
        check_output(git_args + ['checkout', GITHUB_BRANCH])

        log.info('Unstitching output and copying it and readme to working copy')

        fmt_args = dict(name=self.system_name, mentions=MENTIONS_NAME[self.mentions], mapping=self.mapping_id)
        to_add = [os.path.join(root, REFERENCE_OUTPUT_FMT.format(**fmt_args))]
        
        with open(to_add[-1], 'w') as fout:
            print(unstitched_output, file=fout)
        to_add.append(os.path.join(root, REFERENCE_README_FMT.format(**fmt_args)))
        shutil.copyfile(self.readme_path, to_add[-1])
        # TODO evaluate !!

        log.info('Adding new files and committing')
        check_output(git_args + ['add'] + to_add)
        prev_hash = check_output(git_args + ['rev-parse', '--short', 'HEAD']).strip()
        message = 'Output of {name} with {mentions} and {mapping}'.format(**fmt_args)
        check_output(git_args + ['commit', '-m', message])
        new_hash = check_output(git_args + ['rev-parse', '--short', 'HEAD']).strip()
        assert prev_hash != new_hash

        fmt_args['hash'] = new_hash
        branch = '{name}-{hash}'.format(**fmt_args)
        log.info('Pushing to github fork (please authenticate if necessary) as branch {}'.format(branch))
        check_output(git_args + ['push', fork_url, '{}:{}'.format(GITHUB_BRANCH, branch)])

        log.info('Creating pull request')
        print(repo.create_pull(title=message, base=GITHUB_BRANCH, head='{}:{}'.format(self.gh_username, branch)))

        shutil.rmtree(root)
        return 'Done.'

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system_name',
                       help=FIELD_MAP['system_name'].prompt)
        p.add_argument('output_path', nargs='?',
                       help=FIELD_MAP['output_path'].prompt)
        p.add_argument('readme_path', nargs='?',
                       help=FIELD_MAP['readme_path'].prompt)
        p.add_argument('mentions', nargs='?', choices=('system', 'gold', 'linked'),
                       help=FIELD_MAP['mentions'].prompt)
        p.add_argument('mapping_id', nargs='?',
                       help=FIELD_MAP['mapping_id'].prompt)
        p.add_argument('-u', '--gh-username', nargs='?',
                       help=FIELD_MAP['gh_username'].prompt)
        p.add_argument('-p', '--gh-password', nargs='?',
                       help=FIELD_MAP['gh_password'].prompt)
        # TODO: validate mapping_id against repository once more consistently named
        p.set_defaults(cls=cls)
        return p
