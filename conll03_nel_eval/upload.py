"""Tool to deposit a system output as a github pull request"""

import textwrap
import tempfile
import os
from collections import namedtuple
from subprocess import check_output, CalledProcessError
import shutil

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


GOLD_INPUT_NAME = {'all': 'goldmentions',
                   'linked': 'goldlinkedmentions',
                   'system': 'systemmentions'}


Field = namedtuple('Field', 'var prompt desc')

PROMPT_FIELDS = [
    Field('system_name', 'a unique name for your system',
          'This will be used to name the files pertaining to your system in the repository.'),
    Field('output_path', 'the path of your system output on testb (stitched)', ''),
    Field('readme_path', 'the path of a description of your system', ''),
    Field('gold_input', 'whether or not your system uses gold mentions as input (all/linked/system)',
          textwrap.dedent('''
          all: your system receives gold linked and unlinked mentions
          linked: your system only received gold linked mentions
          system: your system generates its own mentions to link''').strip()),
    Field('mapping_id', 'the target ID mapping used (e.g. api20140227)',
          ''),
    Field('gh_username', 'your github username',
          ''),
    Field('gh_password', 'your github password',
          ''),
]


def two_factor_cb(prompt):
    return input(prompt)


class PullRequest(object):
    def __init__(self, system_name=None, output_path=None, readme_path=None,
                 mapping_id=None, gold_input=None, gh_username=None, gh_password=None):
        if github3 is None:
            raise ImportError('Could not import package github3. Please install from PyPI using `pip install github3`')
        try:
            check_output(['git', 'help'])
        except (OSError, CalledProcessError):
            raise OSError('Could not execute a git command. You must have git installed and its commands on PATH.')

        self.gh_username = gh_username
        self.gh = login(gh_username, gh_password, two_factor_callback=two_factor_cb)
        self.system_name = system_name
        self.output_path = output_path
        assert os.path.isfile(output_path)  # TODO: format validation
        self.readme_path = readme_path
        assert os.path.isfile(readme_path)
        self.mapping_id = mapping_id
        self.gold_input = gold_input

        
    def __call__(self):
        repo = self.gh.repository(GITHUB_OWNER, GITHUB_REPO)
        log.info('Creating a personal fork of https://github.com/{}/{}'.format(GITHUB_OWNER, GITHUB_REPO))
        fork = repo.create_fork()
        fork_url = fork.clone_url
        root = tempfile.mkdtemp(suffix='.tmp', prefix='{}-upload-{}')
        # TODO check clone_url includes username, and perhaps consider another form
        log.info('Checking out fork from {} into {}'.format(fork_url, root))
        check_output(['git', 'clone', '--branch', GITHUB_BRANCH, fork_url, root])
        git_args = ['git', '--git-dir', os.path.join(root, '.git'), '--work-tree', root]
        log.info('Unstitching output and copying it and readme to working copy')
        # TODO unstitch !!
        fmt_args = dict(name=self.system_name, mentions=GOLD_INPUT_NAME[self.gold_input], mapping=self.mapping_id)
        to_add = [os.path.join(root, REFERENCE_OUTPUT_FMT.format(**fmt_args))]
        shutil.copyfile(self.output_path, to_add[-1])
        to_add.append(os.path.join(root, REFERENCE_README_FMT.format(**fmt_args)))
        shutil.copyfile(self.readme_path, to_add[-1])
        # TODO evaluate !!

        
        log.info('Adding new files and committing')
        check_output(git_args + ['add'] + to_add)
        prev_hash = check_output(git_args + ['rev-parse', '--short', 'HEAD']).strip()
        message = 'Output of {names} with {mentions} and {mapping}'.format(**fmt_args)
        check_output(git_args + ['commit', '-m', message])
        new_hash = check_output(git_args + ['rev-parse', '--short', 'HEAD']).strip()
        assert prev_hash != new_hash

        fmt_args['hash'] = new_hash
        branch = '{name}-{hash}'.format(**fmt_args)
        log.info('Pushing to github fork (please authenticate if necessary) as branch {}'.format(branch))
        check_output(git_args + ['push', fork_url, ':{}'.format(branch)])

        log.info('Creating pull request')
        fork.create_pull(title=message, base='{}:{}'.format(GITHUB_OWNER, GITHUB_BRANCH), head=branch)

        shutil.rmtree(root)
